import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import collections, transforms, animation, gridspec
from matplotlib import colors as mcolors
import matplotlib.patches as patches
from sort import *

########################
### Helper functions ###
########################

# 2D rotation
def get_2d_R(u):
    return np.array([
        [np.cos(u), -np.sin(u), 0],
        [np.sin(u), np.cos(u), 0],
        [0, 0, 1]
    ])

# phi rotation matrix
def get_R_phi(phi):
    R_phi = np.array(
    [[1, 0, 0],
     [0, np.cos(phi), np.sin(phi)],
     [0, -np.sin(phi), np.cos(phi)]]
    )
    return R_phi.T

# psi rotation matrix
def get_R_psi(psi):
    R_psi = np.array(
    [[np.cos(psi), np.sin(psi), 0],
     [-np.sin(psi), np.cos(psi), 0],
     [0, 0 , 1]]
    )
    return R_psi.T

# derivative phi rotation matrix
def get_R_phi_derivative(phi, phi_dot):
    R_phi = np.array(
    [[0, 0, 0],
     [0, -np.sin(phi), np.cos(phi)],
     [0, -np.cos(phi), -np.sin(phi)]]
    )
    return R_phi.T * phi_dot

# derivative psi rotation matrix, to be multiplied by psi_dot
def get_R_psi_derivative(psi, psi_dot):
    R_psi = np.array(
    [[-np.sin(psi), np.cos(psi), 0],
     [-np.cos(psi), -np.sin(psi), 0],
     [0, 0 , 0]]
    )
    return R_psi.T * psi_dot

# generator for intertial to virtual coordinate transformer
def intertial_to_virtual_coord_gen(psi, phi, f, x_c):
    # phi is tilt, psi is pan
    R_phi = get_R_phi(phi)
    R_psi = get_R_psi(psi)
    def i_to_v_transformer(x):
        q = (R_phi.T @ (R_psi.T @ (x-x_c).reshape(-1,1))).flatten()
        return f*np.array([q[0]/q[2], q[1]/q[2]])
    return i_to_v_transformer

# generator for virtual to inertial coordinate transformer
def virtual_to_inertial_coord(psi, phi, f, x_c):
    R_phi = get_R_phi(phi)
    R_psi = get_R_psi(psi)
    def v_to_i_transformer(p):
        p_aug = (
            R_psi @ R_phi @ (np.array([p[0]/f, p[1]/f, 1])).reshape(-1, 1)
        ).flatten()
        q_z = -x_c[2]/p_aug[2]
        return np.array([p_aug[0]*q_z+x_c[0], p_aug[1]*q_z+x_c[1], 0])
    return v_to_i_transformer

# generator for virtual to inertial coordinate transformer
def virtual_to_inertial_velocity_gen(psi, phi, psi_dot, phi_dot, f, x_c):
    R_phi = get_R_phi(phi)
    R_psi = get_R_psi(psi)
    R_phi_dot = get_R_phi_derivative(phi, phi_dot)
    R_psi_dot = get_R_psi_derivative(psi, psi_dot)
    v_to_i_xf = virtual_to_inertial_coord(psi, phi, f, x_c)
    def v_to_i_vel_transformer(p, p_dot):
        x = v_to_i_xf(p)
        q = (R_phi.T @ (R_psi.T @ (x-x_c).reshape(-1,1))).flatten()
        R_1 = R_psi @ R_phi
        R_2 = R_psi_dot @ R_phi + R_psi @ R_phi_dot
        delta_1 = np.array([p_dot[0]*q[2]/f, p_dot[1]*q[2]/f, 0])
        delta_2 = np.array([q[0]/q[2], q[1]/q[2], 1])
        q_z_dot = -(R_1 @ delta_1.reshape(-1, 1) + R_2 @ q.reshape(-1,1))[2,0]/(R_1 @ delta_2.reshape(-1, 1))[2, 0]
        q_dot = delta_1 + q_z_dot * delta_2
        x_dot = (R_1 @ q_dot.reshape(-1, 1) + R_2 @ q.reshape(-1, 1)).flatten()
        return x_dot
    return v_to_i_vel_transformer

def inertial_velocity_to_psi_phi_dot_gen(psi, phi, f, x_c):
    R_phi = get_R_phi(phi)
    R_psi = get_R_psi(psi)
    R_phi_dot = get_R_phi_derivative(phi, 1)
    R_psi_dot = get_R_psi_derivative(psi, 1)
    x = virtual_to_inertial_coord(psi, phi, f, x_c)([0, 0])
    q_z = (R_phi.T @ (R_psi.T @ (x-x_c).reshape(-1,1))).flatten()[2]
    def inertial_velocity_to_psi_phi_dot_xf(i_vel):
        R_0 = R_psi @ R_phi
        R_1 = R_psi_dot @ R_phi
        R_2 = R_psi @ R_phi_dot
        c_mat = np.zeros((2, 2))
        c_mat[:, 0] = R_1[:, -1][:2] - R_0[:, -1][:2]*R_1[-1, -1]/R_0[-1,-1]
        c_mat[:, 1] = R_2[:, -1][:2] - R_0[:, -1][:2]*R_2[-1, -1]/R_0[-1,-1]
        psi_dot, phi_dot = (np.linalg.inv(c_mat) @ (np.array(i_vel).reshape(-1, 1)/q_z)).flatten() 
        return psi_dot, phi_dot
    return inertial_velocity_to_psi_phi_dot_xf

def process_bboxes_for_sort(boxes, confidences):
    for box in boxes:
        box[2] += box[0]
        box[3] += box[1]
    return np.hstack((np.array(boxes), np.array(confidences).reshape(-1, 1)))

def set_artist_visibility(artist_list, visibility=True):
    for artist in artist_list:
        artist.set_visible(visibility)

########################
### Camera functions ###
########################

# Plot camera
def plot_camera(ax, x_c):
    # Plot camera
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    uc, vc=np.meshgrid(u, v)
    x = x_c[0]+0.5*np.cos(uc)*np.sin(vc)
    y = x_c[1]+0.5*np.sin(uc)*np.sin(vc)
    z = x_c[2]+0.5*np.cos(vc)
    ax.plot_surface(x, y, z, color="darkgray", shade=True)
    ax.text(x_c[0], x_c[1], x_c[2]+2, "PT Camera", fontsize=10)

class CameraFOV:
    
    def __init__(self, ax, psi, phi, psi_dot, phi_dot, f, sensor_w, sensor_h, x_c):
        self.psi = psi
        self.phi = phi
        self.psi_dot = psi_dot
        self.phi_dot = phi_dot
        self.f = f
        self.x_c = x_c
        self.sensor_w = sensor_w
        self.sensor_h = sensor_h
        self.ax = ax
        self.fov_corners, self.fov_collection, self.fov_lines = self._plot_FOV()
        
    def _plot_FOV(self):
        fov_data, fov_verts = self.get_FOV_measurements()
        fov_corners = self.ax.scatter3D(
            fov_data[:, 0], fov_data[:, 1], fov_data[:, 2], depthshade=False, s=1, color="C2", zorder=1)
        fov_collection = Poly3DCollection(fov_verts, facecolors="C2", linewidths=1, edgecolors="C2", alpha=.1, zorder=1)
        self.ax.add_collection3d(fov_collection)
        fov_lines = []
        for i in range(4):
            fov_line, = self.ax.plot3D(
                [self.x_c[0], fov_data[i][0]], 
                [self.x_c[1], fov_data[i][1]], 
                [self.x_c[2], fov_data[i][2]], color="C2", alpha=0.5, lw=1)
            fov_lines.append(fov_line)
        return  fov_corners, fov_collection, fov_lines
        
    def update_FOV(self, psi, phi):
        self.phi = phi
        self.psi = psi
        fov_data, fov_verts = self.get_FOV_measurements()
        self.fov_corners._offsets3d = (fov_data[:, 0], fov_data[:, 1], fov_data[:, 2])
        self.fov_collection.set_verts(fov_verts)
        for i, fov_line in enumerate(self.fov_lines):
            update_data = np.vstack(([self.x_c], fov_data[[i], :])).T
            fov_line.set_data(
                update_data[:2, :]
                )
            fov_line.set_3d_properties(update_data[2, :])
        return self
        
    def get_FOV_measurements(self):
        v_to_i_tx = virtual_to_inertial_coord(self.psi, self.phi, self.f, self.x_c)
        fov_data = np.array([
            v_to_i_tx(np.array([self.sensor_w/2, self.sensor_h/2])),
            v_to_i_tx(np.array([self.sensor_w/2, -self.sensor_h/2])),
            v_to_i_tx(np.array([-self.sensor_w/2, self.sensor_h/2])),
            v_to_i_tx(np.array([-self.sensor_w/2, -self.sensor_h/2]))
        ])
        fov_verts = [ 
            [fov_data[0],fov_data[1],fov_data[3], fov_data[2]]
        ]
        return fov_data, fov_verts

########################
### Object Detector ####
########################
class ObjectDetector:
    
    def __init__(self, ax2, camera_fov, fps=10, dpi=100, threshold=0.25):
        self.ax2 = ax2
        self.extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        self.dpi = dpi
        self.fps = fps
        self.threshold = threshold
        self.camera_fov = camera_fov
        self.sensor_w = self.camera_fov.sensor_w
        self.sensor_h = self.camera_fov.sensor_h
        self.net = cv2.dnn.readNet('yolov2-voc_last.weights', 'yolov2-voc.cfg')
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]
        self.idx = 0
        self.objects = {}
        self.tracker = Sort()
        self.tracking_id = None

    def detect_objects(self):
        self.idx += 1
        # Delete old ax2 patches
        #self.ax2.patches.clear()
        # Save ax2 as image
        if not os.path.exists("tmp/"):
            os.makedirs("tmp")
        image_path = f"tmp/FOV_image_{self.idx}.jpg"
        # Hack to save figure without bounding boxes
        artists_to_exclude = []
        for key in self.objects:
            artists_to_exclude += self.objects[key][:2]
        set_artist_visibility(artists_to_exclude, False)
        self.ax2.figure.savefig(image_path, dpi=self.dpi, bbox_inches=self.extent)
        set_artist_visibility(artists_to_exclude, True)
        # Get bounding boxes with YOLO model
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        height, width, channels = frame.shape
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                confidence = detection[5]
                if confidence > 0.6:
                    center_x = (detection[0]-1/2) * self.sensor_w
                    center_y = (1/2-detection[1]) * self.sensor_h
                    w = detection[2] * self.sensor_w
                    h = detection[3] * self.sensor_h
                    x = center_x - w / 2
                    y = center_y - h / 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        # Update tracker
        if np.array(boxes).size != 0:
            # Get boxes for sort
            sort_boxes = process_bboxes_for_sort(boxes, confidences)
            track_bbs_ids = self.tracker.update(sort_boxes)
        else:
            track_bbs_ids = self.tracker.update()
        # Update objects dictionary, bounding boxes and text
        for i, bbox_data in enumerate(track_bbs_ids):
            bbox_id = int(bbox_data[-1])
            bbox_coords =  bbox_data[:-1]
            x, y, w, h = bbox_coords[0], bbox_coords[1], bbox_coords[2] - bbox_coords[0], bbox_coords[3] - bbox_coords[1]
            confidence = 0
            trk_idx = 0 # This might cause issues
            for i, trk in enumerate(self.tracker.trackers):
                if trk.id == bbox_id-1:
                    trk_idx = i
                    confidence = trk.conf
            if bbox_id not in self.objects:
                self.objects[bbox_id] = []
                rect = patches.Rectangle((x, y),
                                         w, h, 
                                         linewidth=1, edgecolor='green', facecolor='none')
                self.ax2.add_patch(rect)
                self.objects[bbox_id].append(rect)
                textbox = self.ax2.text(x, y+h+0.01, 
                                   f"Object:{confidence:0.2f}, ID:{int(bbox_id)}", color='black', 
                                   fontsize=7)
                textbox.set_bbox(dict(facecolor='green', pad=0.1, edgecolor='green'))
                self.objects[bbox_id].append(textbox)
                # Append inertial x, y, and initial velocity = 0
                # vel_est, num detections, num(velocity>threshold), average vel_est
                self.objects[bbox_id] += [0, 0, 0, 0, 0, 0, []]
            else:
                v_to_i_xf = virtual_to_inertial_coord(
                    self.camera_fov.psi, self.camera_fov.phi, 
                    self.camera_fov.f, self.camera_fov.x_c)
                self.objects[bbox_id][8].append(v_to_i_xf([x, y])[:2])
                if len(self.objects[bbox_id][8]) >=2:
                    xy_array = np.array(self.objects[bbox_id][8])
                    t = np.arange(len(xy_array))
                    vel_est_alt_x = np.sum((xy_array[:, 0]-xy_array[:, 0].mean())*(t-t.mean()))/np.sum((t-t.mean())**2)
                    vel_est_alt_y = np.sum((xy_array[:, 1]-xy_array[:, 1].mean())*(t-t.mean()))/np.sum((t-t.mean())**2)
                #p_x 
                x_dot = self.tracker.trackers[trk_idx].kf.x[4][0]
                #p_y
                y_dot = self.tracker.trackers[trk_idx].kf.x[5][0]
                vel_est = virtual_to_inertial_velocity_gen(
                            self.camera_fov.psi, self.camera_fov.phi, 
                            self.camera_fov.psi_dot/self.fps, self.camera_fov.phi_dot/self.fps, 
                            self.camera_fov.f, self.camera_fov.x_c)([x, y], [x_dot, y_dot])
                if len(self.objects[bbox_id][8]) >=3:
                    vel_est = [vel_est_alt_x, vel_est_alt_y]
                vel_est_norm = np.linalg.norm(vel_est)
                self.objects[bbox_id][2] = vel_est[0]
                self.objects[bbox_id][3] = vel_est[1]
                self.objects[bbox_id][4] += 1
                self.objects[bbox_id][5] += (vel_est_norm >= self.threshold*10/self.fps)*1
                # Keep track of time average velocities
                n_dets = self.objects[bbox_id][4]
                self.objects[bbox_id][6] = self.objects[bbox_id][6]*(n_dets-1)/(n_dets) + vel_est[0]/n_dets
                self.objects[bbox_id][7] = self.objects[bbox_id][7]*(n_dets-1)/(n_dets) + vel_est[1]/n_dets
                # Update artists
                rect = self.objects[bbox_id][0]
                textbox = self.objects[bbox_id][1]
                textbox.set_position((x, y+h+0.01))
                rect.set_xy((x, y))
                rect.set_width(w)
                rect.set_height(h)
                proba_target = self.objects[bbox_id][5]/self.objects[bbox_id][4]
                if proba_target > 0.5:
                    textbox.set_text(f"Target:{confidence:0.2f}, ID:{int(bbox_id)}")
                    textbox.set_bbox(dict(facecolor='red', pad=0.1, edgecolor='red'))
                    rect.set_edgecolor('red')
                    if self.objects[bbox_id][4]>=2 and self.tracking_id is None:
                        self.tracking_id = int(bbox_id)
                else:
                    textbox.set_text(f"Object:{confidence:0.2f}, ID:{int(bbox_id)}")
                    textbox.set_bbox(dict(facecolor='green', pad=0.1, edgecolor='green'))
                    rect.set_edgecolor('green')
        # Cleanup objects dict
        for key in list(self.objects):
            if key not in track_bbs_ids[:, -1]:
                rect = self.objects[key][0]
                textbox = self.objects[key][1]
                rect.remove()
                textbox.remove()
                self.objects.pop(key)
                if self.tracking_id == key:
                    self.tracking_id = None


#########################
### Vehicle functions ###
#########################
class LinearVehicle:
    
    def __init__(self, ax, ax2, v_l=3, v_w=3, x_offset=0, y_offset=0, vel_dir=[1, 1], color="C0", is_visible=True):
        self.ax = ax
        self.ax2 = ax2
        self.v_l = v_l
        self.v_w = v_w
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.vel_dir = vel_dir
        self.color = color
        # Initialize vehicles
        #self.v_vel = np.random.uniform(0.5, 1.5) * np.array(vel_dir)
        self.v_vel = np.array(vel_dir)
        self.v_data_0, self.v_corners, self.v_collection = self._plot_vehicle()
        self.vv_corners, self.vv_collection = None, None
        self.is_visible = is_visible
        if not self.is_visible:
            self.set_visible(False)
            
    def _plot_vehicle(self):
        # Inertial dataframe
        v_data_init = np.array([[self.v_w, -self.v_l/2, 0], 
                   [self.v_w, self.v_l/2, 0], 
                   [0, -self.v_l/2, 0], 
                   [0, +self.v_l/2, 0]])
        R_mat = get_2d_R(np.arctan(self.v_vel[1]/self.v_vel[0]))
        v_data = (R_mat @ v_data_init.T).T
        v_data[:, 0] += self.x_offset
        v_data[:, 1] += self.y_offset
        v_corners = self.ax.scatter3D(
            v_data[:, 0], v_data[:, 1], v_data[:, 2], depthshade=False, s=1, color=self.color, zorder=10)
        v_verts = [ 
            [v_data[0],v_data[1],v_data[3], v_data[2]]
        ]
        v_collection = Poly3DCollection(v_verts, 
         facecolors=self.color, linewidths=1, edgecolors=self.color, alpha=.5, zorder=10)
        self.ax.add_collection3d(v_collection)
        return v_data, v_corners, v_collection
    
    def update_vehicle(self, t):
        v_data = self.get_vehicle_data(t)
        v_verts = [ 
                [v_data[0],v_data[1],v_data[3], v_data[2]]
            ]
        self.v_corners._offsets3d = (v_data[:, 0], v_data[:, 1], v_data[:, 2])
        self.v_collection.set_verts(v_verts)
        return self
    
    def update_vehicle_virtual(self, camera_fov, t):
        vv_data = self.get_vehicle_data_virtual(camera_fov, t)
        vv_verts = [ 
                [vv_data[0],vv_data[1],vv_data[3], vv_data[2]]
            ]
        if self.vv_corners is None:
            self.vv_corners = self.ax2.scatter(vv_data[:, 0], vv_data[:, 1], color=self.color, s=1)
            self.vv_collection = collections.PolyCollection(
                vv_verts, 
                facecolors=self.color, linewidths=1, edgecolors=self.color, alpha=.5)
            self.ax2.add_collection(self.vv_collection, autolim=True)
        else:
            self.vv_corners.set_offsets(vv_data)
            self.vv_collection.set_verts(vv_verts)
        return self
    
    def get_vehicle_data(self, t):
        v_data = np.zeros((4, 3))
        v_data[:, 0] = self.v_data_0[:, 0] + self.v_vel[0]*t
        v_data[:, 1] = self.v_data_0[:, 1] + self.v_vel[1]*t
        return v_data
    
    def get_vehicle_data_virtual(self, camera_fov, t):
        v_data = self.get_vehicle_data(t)
        psi, phi, f, x_c = camera_fov.psi, camera_fov.phi, camera_fov.f, camera_fov.x_c
        i_to_v_tx = intertial_to_virtual_coord_gen(psi, phi, f, x_c)
        vv_data = np.zeros((4, 2))
        for i, vec in enumerate(v_data):
            vv_data[i] = i_to_v_tx(v_data[i])
        return vv_data

    def set_visible(self, visibility=False):
        self.v_collection.set_visible(visibility)
        self.v_corners.set_visible(visibility)
        if self.vv_corners is not None and self.vv_collection is not None:
            self.vv_corners.set_visible(visibility)
            self.vv_collection.set_visible(visibility)
        self.is_visible = visibility
        return self


############################
### Background functions ###
############################
class BackgroundObject:
    
    def __init__(self, ax, ax2, v_l=3, v_w=3, x_offset=0, y_offset=0, vel_dir=[1, 1], color="C0"):
        self.ax = ax
        self.ax2 = ax2
        self.v_l = v_l
        self.v_w = v_w
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.vel_dir = vel_dir
        self.color = color
        # Initialize vehicles
        self.v_data_0, self.v_corners, self.v_collection = self._plot_vehicle()
        self.vv_corners, self.vv_collection = None, None
    
    def _plot_vehicle(self):
        # Inertial dataframe
        v_data_init = np.array([[self.v_w, -self.v_l/2, 0], 
                   [self.v_w, self.v_l/2, 0], 
                   [0, -self.v_l/2, 0], 
                   [0, +self.v_l/2, 0]])
        if self.vel_dir[0]==0:
             R_mat = get_2d_R(np.pi/2)
        else:
            R_mat = get_2d_R(np.arctan(self.vel_dir[1]/self.vel_dir[0]))
        v_data = (R_mat @ v_data_init.T).T
        v_data[:, 0] += self.x_offset
        v_data[:, 1] += self.y_offset
        v_corners = self.ax.scatter3D(
            v_data[:, 0], v_data[:, 1], v_data[:, 2], depthshade=False, s=1, color=self.color, zorder=5)
        v_verts = [ 
            [v_data[0],v_data[1],v_data[3], v_data[2]]
        ]
        v_collection = Poly3DCollection(v_verts, 
         facecolors=self.color, linewidths=1, edgecolors=self.color, alpha=.5, zorder=5)
        self.ax.add_collection3d(v_collection)

        return v_data, v_corners, v_collection
    
    def update_vehicle_virtual(self, camera_fov):
        vv_data = self.get_vehicle_data_virtual(camera_fov)
        vv_verts = [ 
                [vv_data[0],vv_data[1],vv_data[3], vv_data[2]]
            ]
        if self.vv_corners is None:
            self.vv_corners = self.ax2.scatter(vv_data[:, 0], vv_data[:, 1], color=self.color, s=1)
            self.vv_collection = collections.PolyCollection(
                vv_verts, 
                facecolors=self.color, linewidths=1, edgecolors=self.color, alpha=.5)
            self.ax2.add_collection(self.vv_collection, autolim=True)
        else:
            self.vv_corners.set_offsets(vv_data)
            self.vv_collection.set_verts(vv_verts)
        return self
    
    def get_vehicle_data_virtual(self, camera_fov):
        psi, phi, f, x_c = camera_fov.psi, camera_fov.phi, camera_fov.f, camera_fov.x_c
        i_to_v_tx = intertial_to_virtual_coord_gen(psi, phi, f, x_c)
        vv_data = np.zeros((4, 2))
        for i, vec in enumerate(self.v_data_0):
            vv_data[i] = i_to_v_tx(self.v_data_0[i])
        return vv_data


#######################
### Animation Setup ###
#######################

def animation_setup(sensor_w, sensor_h, f, x_c, psi_0, phi_0, t_max, fps):
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(4,2)

    ax = fig.add_subplot(gs[:, 0], projection='3d')
    #ax.w_zaxis.set_pane_color(mcolors.to_rgba('whitesmoke'))
    ax.set_box_aspect([2,2,1])
    ax2 = fig.add_subplot(gs[:2, 1])

    ax3 = fig.add_subplot(gs[2, 1])
    ax4 = fig.add_subplot(gs[3, 1])
    fig.tight_layout(pad=1.5, w_pad=5)
    fig.subplots_adjust(right=0.93)

    x_max = 40

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, x_max)
    ax.set_zlim(0, x_max/2)
    ax.set_zticks([0,5,10,15,20])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax2.set_yticks([-2, -1, 0, 1, 2])
    ax2.set_xlim(-sensor_w/2, sensor_w/2)
    ax2.set_ylim(-sensor_h/2, sensor_h/2)
    #ax2.set_facecolor('whitesmoke')
    ax2.set_title("Virtual FOV: Surveillance")
    ax3.set_title("Input Voltages (normalized) vs Time", fontsize=10)
    ax3.set_ylim(-1.02, 1.02)
    ax3.set_yticks([-1, 0, 1])
    ax4.set_title("Pan and Tilt Angles vs Time", fontsize=10)

    # Set up angle plot
    ax4.set_xlim(0, t_max)
    ax4.set_ylim(-0.02, 2*np.pi+0.02)
    ax4.set_yticks([0, np.pi*0.5, np.pi, np.pi*3/2, 2*np.pi], 
                   [0, r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
    ax4.set_ylabel(r'$\psi_t$', color = "C0") 
    ax4.tick_params(axis ='y', labelcolor = "C0")
    psi_line, = ax4.plot(
                    [], 
                    [], 
                    color="C0")
    psi_data = np.zeros((2, t_max*fps))
    psi_data[0] = np.arange(t_max*fps)/fps

    ax4.set_xlabel("time (s)")
    ax4_phi = ax4.twinx() 
    ax4_phi.set_ylim(-0.02+np.pi/2, np.pi+0.02)
    ax4_phi.set_ylabel(r'$\phi_t$', color = "C3") 
    ax4_phi.tick_params(axis ='y', labelcolor = "C3") 
    ax4_phi.set_yticks([np.pi*0.5, np.pi*0.75, np.pi], 
                       [r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]) 
    phi_line, = ax4_phi.plot(
                    [], 
                    [], 
                    color="C3")
    phi_data = np.zeros((2, t_max*fps))
    phi_data[0] = np.arange(t_max*fps)/fps

    # Set up u plots
    ax3.set_xlim(0, t_max)
    ax3.set_ylabel(r'$u_1$', color = "C0") 
    ax3.tick_params(axis ='y', labelcolor = "C0")
    u1_line, = ax3.plot(
                    [], 
                    [], 
                    color="C0")
    u1_data = np.zeros((2, t_max*fps))
    u1_data[0] = np.arange(t_max*fps)/fps

    ax3_phi = ax3.twinx() 
    ax3_phi.set_ylabel(r'$u_2$', color = "C3") 
    ax3_phi.set_ylim(-1.02, 1.02)
    ax3_phi.tick_params(axis ='y', labelcolor = "C3") 
    u2_line, = ax3_phi.plot(
                    [], 
                    [], 
                    color="C3")
    u2_data = np.zeros((2, t_max*fps))
    u2_data[0] = np.arange(t_max*fps)/fps
    # Plot camera
    plot_camera(ax, x_c)

    # Add camera FOV
    camera_fov = CameraFOV(
        ax=ax,
        psi=psi_0, phi=phi_0, 
        psi_dot=0, phi_dot=0,
        f=f, sensor_w=sensor_w, sensor_h=sensor_h, x_c=x_c)
    
    return fig, ax, ax2, camera_fov, psi_line, psi_data, phi_line, phi_data, u1_line, u1_data, u2_line, u2_data