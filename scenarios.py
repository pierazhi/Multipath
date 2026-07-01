import plotly.graph_objects as go
import trimesh
import numpy as np
from campaign import *
from cir import *
from constants import *
from diffraction import *
from em_core import *
from engine_hybrid import *
from engine_image import *
from engine_image_parallel import *
from engine_sbr import *
from geometry import *
from legacy import *
from terrain_io import *
from timing import *
from viz import *
import plotly.io as pio

BASE_DIR = Path(__file__).resolve().parent

tifs_dir = BASE_DIR / "tifs_new"
kernels_dir = BASE_DIR / "kernels"
meshes_dir = BASE_DIR / "meshes"


def choose_scenario(case):
    if case == "lunar":
        mesh = generate_lunar_mesh_2(
            size=1000.0,          
            triangle_resolution=10.0,        
            terrain_type='basin', 
            num_craters=50, 
            min_crater_r=3.0, 
            max_crater_r=100.0,
            seed=42
        )    
        bounds = mesh.bounds
        x_min, y_min = bounds[0][0], bounds[0][1]
        x_max, y_max = bounds[1][0], bounds[1][1]

        # center_tx = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2]) + np.array([1e6, 1e6])
        center_rx = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])

        pos_tx = np.array([231.807, 665.9274, 274.005])
        pos_rx = np.array([43.8012, -24.3837, -121.1798])

        # pos_tx = np.array([9979847.091,2740434.545, 11632614.997])
        # pos_rx = np.array([43.8012, -24.3837, -121.1798])

        # pos_tx = np.asarray(generate_grid_nodes(mesh, center_tx, n=5, spacing=250, height=1e6, 
        #                     mode='plane', bounded=False)).squeeze()
        # pos_rx = np.asarray(generate_grid_nodes(mesh, center_rx, n=1, spacing=30, height=5, 
        #                     mode='terrain', bounded=True)).squeeze()
        
        # pos_tx = np.asarray(generate_grids(mesh, center=center_tx, n=1, spacing=50, height=1e3, bounded=False)).squeeze()
        # pos_rx = np.asarray(generate_grids(mesh, center=center_rx, n=1, spacing=30, height=10)).squeeze()
        
        # i, j = 5, 0
        # pos_tx = pos_tx[i, :]
        # pos_rx = pos_rx[j, :]

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

        sep = np.linalg.norm(pos_rx - pos_tx)


    elif case == "ground":
        v0 = [1.0, -1.0, 0.0]
        v1 = [1.0, 1.0, 0.0]
        v2 = [-1.0, 1.0, 0.0]
        v3 = [-1.0, -1.0, 0.0]

        vertices = np.array([v0, v1, v2, v3])
        # Definizione di tutte le facce (comprese quelle della Sezione 2)
        faces = np.array([
            # Sezione 1
            [0, 1, 2], [0, 2, 3], # Muro fisso 
        ])

        # Generazione della mesh finale
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        origin = np.zeros(3)
        pos_tx = np.array([-5, 0, 5])
        pos_rx = np.array([5, 0, 5]) 

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

    elif case == "wall":
        v0 = [1.0, 0.0, 1.0]
        v1 = [-1.0, 0.0, 1.0]
        v2 = [-1.0, 0.0, -1.0]
        v3 = [1.0, 0.0, -1.0]

        vertices = np.array([v0, v1, v2, v3])
        # Definizione di tutte le facce (comprese quelle della Sezione 2)
        faces = np.array([
            # Sezione 1
            [0, 1, 2], [0, 2, 3], # Muro fisso 
        ])

        # Generazione della mesh finale
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        origin = np.zeros(3)
        pos_tx = np.array([0, -5, 5])
        pos_rx = np.array([0, -5, -5]) 

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

    elif case == "corridoio":
        angle_deg = 0  
        angle_rad1 = np.radians(angle_deg + 90)
        angle_rad2 = np.radians(angle_deg + 90)

        # --- SEZIONE L 1 (Fissa/Base) ---
        v0 = [0.0, 0.0, 0.0]
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 1.0]
        v3 = [0.0, 0.0, 1.0]

        x_orig, y_orig = 0.0, 1.0
        x_rot1 = x_orig * np.cos(angle_rad1) - y_orig * np.sin(angle_rad1)
        y_rot1 = x_orig * np.sin(angle_rad1) + y_orig * np.cos(angle_rad1)

        v4 = [x_rot1, y_rot1, 0.0]
        v5 = [x_rot1, y_rot1, 1.0]

        shift = -1.0
        v6 = [0.0, 0.0 + shift, 0.0]
        v7 = [1.0, 0.0 + shift, 0.0]
        v8 = [1.0, 0.0 + shift, 1.0]
        v9 = [0.0, 0.0 + shift, 1.0]

        x_rot2 = x_orig * np.cos(angle_rad2) - y_orig * np.sin(angle_rad2)
        y_rot2 = x_orig * np.sin(angle_rad2) + y_orig * np.cos(angle_rad2)

        v10 = [x_rot2, y_rot2 + shift, 0.0]
        v11 = [x_rot2, y_rot2 + shift, 1.0]

        vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11])
        faces = np.array([
            # Sezione 1
            [0, 1, 2], [0, 2, 3], # Muro fisso 1
            [0, 3, 5], [0, 5, 4], # Muro ruotato 1 (0°)
            
            # Sezione 2
            [6, 7, 8], [6, 8, 9], # Muro fisso 2
            [6, 9, 11], [6, 11, 10] # Muro ruotato 2 (180°)
        ])

            # Generazione della mesh finale
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        pos_tx = np.array([-1, -0.5, 0.5])
        pos_rx = np.array([1.0, -0.5, 0.5]) 

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

    elif case == "triangoli_angolati":
        # Parametri geometrici configurabili
        pitch_angle = np.radians(45)  # Inclinazione rispetto al piano XY
        spacing = 3.0                 # Distanza totale tra i centri lungo X
        offset_x = spacing / 2.0
        z_clearance = 2.0             # Distanza in Z di TX e RX sopra i centroidi

        # Vertici di base di un triangolo nel piano YZ (centrato nell'origine)
        # Base sull'asse Y, altezza lungo l'asse Z
        v0_local = np.array([0.0, -1.0, 0.0])
        v1_local = np.array([0.0,  1.0, 0.0])
        v2_local = np.array([0.0,  0.0, 2.0])

        # Funzione di utilità per la matrice di rotazione attorno all'asse Y
        def rot_y(theta):
            return np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])

        # --- Triangolo 1 (A destra, inclinato verso il centro) ---
        # Ruotiamo di +pitch_angle e trasliamo a +offset_x
        R1 = rot_y(pitch_angle)
        v0 = R1 @ v0_local + np.array([offset_x, 0.0, 0.0])
        v1 = R1 @ v1_local + np.array([offset_x, 0.0, 0.0])
        v2 = R1 @ v2_local + np.array([offset_x, 0.0, 0.0])

        # --- Triangolo 2 (A sinistra, inclinato verso il centro) ---
        # Ruotiamo di -pitch_angle e trasliamo a -offset_x
        R2 = rot_y(-pitch_angle)
        v3 = R2 @ v0_local + np.array([-offset_x, 0.0, 0.0])
        v4 = R2 @ v1_local + np.array([-offset_x, 0.0, 0.0])
        v5 = R2 @ v2_local + np.array([-offset_x, 0.0, 0.0])

        vertices = np.array([v0, v1, v2, v3, v4, v5])
        
        # Facce orientate in modo che le normali puntino verso lo spazio interno
        faces = np.array([
            [0, 2, 1],  # Triangolo 1 (A destra, guarda verso l'asse -X)
            [3, 4, 5]   # Triangolo 2 (A sinistra, guarda verso l'asse +X)
        ])

        # Generazione della mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # --- Calcolo dei centroidi e posizionamento TX/RX ---
        centroid_1 = np.mean([v0, v1, v2], axis=0)
        centroid_2 = np.mean([v3, v4, v5], axis=0)

        # Posizioniamo TX e RX in alto (+Z) esattamente sopra i centroidi
        pos_tx = centroid_2 + np.array([0.0, 0.0, z_clearance])
        pos_rx = centroid_1 + np.array([0.0, 0.0, z_clearance])

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

        sep = np.linalg.norm(pos_tx - pos_rx)

    elif case == "angolo_diedro":
        # Due pareti ortogonali a 90 gradi tra loro
        # Parete 1 (Piano YZ, a X=0)
        v0 = [0.0, 0.0, 0.0]
        v1 = [0.0, 2.0, 0.0]
        v2 = [0.0, 2.0, 2.0]
        v3 = [0.0, 0.0, 2.0]

        # Parete 2 (Piano XZ, a Y=0)
        v4 = [0.0, 0.0, 0.0]
        v5 = [2.0, 0.0, 0.0]
        v6 = [2.0, 0.0, 2.0]
        v7 = [0.0, 0.0, 2.0]

        vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
        faces_list = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6)]
        faces = np.array(faces_list)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # TX e RX posizionati davanti all'angolo per catturare il ritorno retro-riflesso
        pos_tx = np.array([1.5, 1.5, 1.0])
        pos_rx = np.array([1.2, 1.2, 1.0])

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

    elif case == "cavita_parabolica":
        # Creiamo una forma a C o semi-ottagono usando triangoli verticali
        # Definiamo i punti della base sul piano XY
        p0 = [ 2.0,  0.0, 0.0]
        p1 = [ 1.4,  1.4, 0.0]
        p2 = [ 0.0,  2.0, 0.0]
        p3 = [-1.4,  1.4, 0.0]
        p4 = [-2.0,  0.0, 0.0]
        
        # Punti superiori alla quota Z = 2.0
        p0_h, p1_h, p2_h, p3_h, p4_h = [p[:2] + [2.0] for p in [p0, p1, p2, p3, p4]]

        vertices = np.array([p0, p1, p2, p3, p4, p0_h, p1_h, p2_h, p3_h, p4_h])
        faces_list = [
            (0, 5, 6), (0, 6, 1),  # Pannello 1
            (1, 6, 7), (1, 7, 2),  # Pannello 2
            (2, 7, 8), (2, 8, 3),  # Pannello 3
            (3, 8, 9), (3, 9, 4)   # Pannello 4
        ]
        faces = np.array(faces_list)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # TX al centro della parabola, RX nel "fuoco" geometrico per ricevere tutti i rimbalzi
        pos_tx = np.array([0.0, 0.5, 1.0])
        pos_rx = np.array([0.0, 1.2, 1.0])

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

    elif case == "prisma_chiuso":
        # Vertici di un triangolo equilatero sul piano XY, estesi in Z
        # Lato del prisma = ~3.46 unità
        v0 = [ 0.0,  2.0, 0.0]
        v1 = [ 1.73, -1.0, 0.0]
        v2 = [-1.73, -1.0, 0.0]
        
        v3 = [ 0.0,  2.0, 3.0]
        v4 = [ 1.73, -1.0, 3.0]
        v5 = [-1.73, -1.0, 3.0]

        vertices = np.array([v0, v1, v2, v3, v4, v5])
        faces_list = [
            (0, 1, 4), (0, 4, 3),  # Parete 1
            (1, 2, 5), (1, 5, 4),  # Parete 2
            (2, 0, 3), (2, 3, 5)   # Parete 3
        ]
        faces = np.array(faces_list)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # TX e RX sono intrappolati dentro il prisma
        pos_tx = np.array([0.0, 0.0, 1.0])
        pos_rx = np.array([0.2, 0.2, 1.5])

    elif case == "biliardo":
        # Dimensioni del tavolo da biliardo (Area di gioco interna)
        length = 100 # asse X
        width = 50   # asse Y
        wall_height = 10
        wall_thick = 0.1

        # 1. Creiamo le 4 sponde (Muri)
        # Usiamo traslazioni per posizionarle attorno all'area di gioco
        cushion_bottom = trimesh.creation.box(
            extents=[length + 2*wall_thick, wall_thick, wall_height],
            transform=trimesh.transformations.translation_matrix([length/2, -wall_thick/2, wall_height/2])
        )
        cushion_top = trimesh.creation.box(
            extents=[length + 2*wall_thick, wall_thick, wall_height],
            transform=trimesh.transformations.translation_matrix([length/2, width + wall_thick/2, wall_height/2])
        )
        cushion_left = trimesh.creation.box(
            extents=[wall_thick, width, wall_height],
            transform=trimesh.transformations.translation_matrix([-wall_thick/2, width/2, wall_height/2])
        )
        cushion_right = trimesh.creation.box(
            extents=[wall_thick, width, wall_height],
            transform=trimesh.transformations.translation_matrix([length + wall_thick/2, width/2, wall_height/2])
        )

        # 2. Creiamo il tavolo (Il panno verde)
        table_bed = trimesh.creation.box(
            extents=[length, width, 0.05],
            transform=trimesh.transformations.translation_matrix([length/2, width/2, -0.025])
        )

        # 3. Ostacolo Centrale (Blocca la Line-of-Sight diretta tra TX e RX)
        center_bumper = trimesh.creation.cylinder(
            radius=10, 
            height=wall_height,
            transform=trimesh.transformations.translation_matrix([length/2, width/2, wall_height/2])
        )

        # 4. Concateniamo in un'unica mesh
        mesh = trimesh.util.concatenate([
            cushion_bottom, 
            cushion_top, 
            cushion_left, 
            cushion_right, 
            table_bed, 
            center_bumper
        ])

        # TX posizionato nell'angolo in basso a sinistra (leggermente sollevato dal fondo)
        pos_tx = np.array([0.05*length, 0.5*width, 0.05])
        
        # RX posizionato nell'angolo in alto a destra
        pos_rx = np.array([0.95*length, 0.5*width, 0.05])

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

        sep = np.linalg.norm(pos_rx - pos_tx)

    elif case == "polarization_check":
        pitch_deg, roll_deg, yaw_deg = 45, 0, 0
        v1_tri1 = [ 1.0, -2.5, 0.0]
        v2_tri1 = [ 1.0,  2.5, 0.0]
        v3_tri1 = [-2.0,  0.0, 0.0] 
        vertices = np.array([v1_tri1, v2_tri1, v3_tri1])
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        apply_rpy(mesh, pitch_deg, roll_deg, yaw_deg)

        centroid = mesh.centroid
        normal = np.asarray(mesh.face_normals).squeeze()
        quota_sopra = 5.0
        pos_tx = centroid + np.array([0.0, 0.0, quota_sopra])  

        ray_vector = centroid - pos_tx
        k_in = ray_vector / np.linalg.norm(ray_vector)  # Normalize it
        k_out = k_in - 2 * np.dot(k_in, normal) * normal

        rx_distance = 5.0
        pos_rx = centroid + (k_out * rx_distance)

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

    elif case == "campaign":
        mesh = generate_lunar_mesh_2(
            size=1000.0,          
            triangle_resolution = 10.0,        
            terrain_type='basin', 
            num_craters=25, 
            min_crater_r=1.0, 
            max_crater_r=100.0,
            seed=42
        )    
        bounds = mesh.bounds
        x_min, y_min = bounds[0][0], bounds[0][1]
        x_max, y_max = bounds[1][0], bounds[1][1]

        center_tx = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2]) - np.array([1, 0.35])*1e3
        center_rx = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])

        pos_tx = np.asarray(generate_grid_nodes(mesh, center_tx, n=5, spacing=100, height=250, 
                            mode='plane', bounded=False)).squeeze()
        
        pos_rx = np.asarray(generate_grid_nodes(mesh, center_rx, n=5, spacing=30, height=5, 
                            mode='terrain', bounded=True)).squeeze()

        # pos_tx = np.array([231.807, 665.9274, 274.005])
        # pos_rx = np.array([43.8012, -24.3837, -131.1798])

        # pos_tx = np.asarray(generate_grids(mesh, center_tx, n=3, spacing=100, height=1e6)).squeeze()
        # pos_rx = np.asarray(generate_grids(mesh, center_rx, n=3, spacing=30, height=1.2)).squeeze()


    elif case == "canyon":
        path = r"/Users/pierazhi/Desktop/Multipath/Multipath/dem2mesh/meshes/simple_canyon.ply"
        # path = r"C:\Users\Luna\Documents\MP\Multipath\dem2mesh\meshes\simple_canyon.ply"

        mesh = trimesh.load_mesh(path)
        pos_tx = np.array([50, 10, 100])
        pos_rx = np.array([0, 0, 2])

        # pos_tx = np.array([0, 0, 10])
        # pos_rx = np.array([100,0,10])

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX

        sep = np.linalg.norm(np.asarray(pos_tx) - np.asarray(pos_rx))

    elif case == "florence":
        # path = r"/Users/pierazhi/Desktop/Multipath/Multipath/dem2mesh/meshes/florence.ply"
        path = r"/Users/pierazhi/Desktop/Multipath/Multipath/dem2mesh/meshes/florence.ply"

        mesh = trimesh.load_mesh(path)

        pos_tx = np.array([-101.46, -192.12, 28.19])
        pos_rx = np.array([-0.11, 0.49, 2.02])

        sep = np.linalg.norm(np.asarray(pos_tx) - np.asarray(pos_rx))

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX


    elif case == "real_lunar":
        file = r"/Users/pierazhi/Desktop/tifs/LDEM_20M_clip.tif"

        _, _, _, _, mesh = tif2mesh(file) 
        bounds = mesh.bounds
        x_min, y_min, z_min = bounds[0][0], bounds[0][1], bounds[0][2]
        x_max, y_max, z_max = bounds[1][0], bounds[1][1], bounds[1][2]

        # center_tx = np.array([(x_max + x_min) / 2, (y_max + y_min) / 2])
        center_tx = np.array([-12229, 12190])
        center_rx = np.array([-10 + (x_max + x_min) / 2, (y_max + y_min) / 2])

        pos_rx = np.asarray(generate_grid_nodes(mesh, center_rx, n=3, spacing=100, height=10, 
                        mode='terrain', bounded=True)).squeeze()
        pos_tx = np.asarray(generate_grid_nodes(mesh, center_tx, n=3, spacing=30, height=-1900, 
                        mode='plane', bounded=False)).squeeze()
        
        i, j = 0, 0
        pos_tx = pos_tx[i, :]
        pos_rx = pos_rx[j, :]

        boresight_tx = normalize(pos_rx - pos_tx) # Il TX punta verso l'RX
        boresight_rx = normalize(pos_tx - pos_rx) # L'RX punta verso il TX
    
        sep = np.linalg.norm(np.asarray(pos_tx) - np.asarray(pos_rx))

    return mesh, pos_tx, pos_rx, boresight_tx, boresight_rx, sep


def choose_mesh(case):
    if case == "lunar":
        mesh = generate_lunar_mesh_2(
            size=1000.0,          
            triangle_resolution=10.0,        
            terrain_type='basin', 
            num_craters=50, 
            min_crater_r=3.0, 
            max_crater_r=100.0,
            seed=42
        )    

    elif case == "biliardo":
        # Dimensioni del tavolo da biliardo (Area di gioco interna)
        length = 100 # asse X
        width = 50   # asse Y
        wall_height = 10
        wall_thick = 0.1

        # 1. Creiamo le 4 sponde (Muri)
        # Usiamo traslazioni per posizionarle attorno all'area di gioco
        cushion_bottom = trimesh.creation.box(
            extents=[length + 2*wall_thick, wall_thick, wall_height],
            transform=trimesh.transformations.translation_matrix([length/2, -wall_thick/2, wall_height/2])
        )
        cushion_top = trimesh.creation.box(
            extents=[length + 2*wall_thick, wall_thick, wall_height],
            transform=trimesh.transformations.translation_matrix([length/2, width + wall_thick/2, wall_height/2])
        )
        cushion_left = trimesh.creation.box(
            extents=[wall_thick, width, wall_height],
            transform=trimesh.transformations.translation_matrix([-wall_thick/2, width/2, wall_height/2])
        )
        cushion_right = trimesh.creation.box(
            extents=[wall_thick, width, wall_height],
            transform=trimesh.transformations.translation_matrix([length + wall_thick/2, width/2, wall_height/2])
        )

        # 2. Creiamo il tavolo (Il panno verde)
        table_bed = trimesh.creation.box(
            extents=[length, width, 0.05],
            transform=trimesh.transformations.translation_matrix([length/2, width/2, -0.025])
        )

        # 3. Ostacolo Centrale (Blocca la Line-of-Sight diretta tra TX e RX)
        center_bumper = trimesh.creation.cylinder(
            radius=10, 
            height=wall_height,
            transform=trimesh.transformations.translation_matrix([length/2, width/2, wall_height/2])
        )

        # 4. Concateniamo in un'unica mesh
        mesh = trimesh.util.concatenate([
            cushion_bottom, 
            cushion_top, 
            cushion_left, 
            cushion_right, 
            table_bed, 
            center_bumper
        ])

    elif case == "ground":
        v0 = [1.0, -1.0, 0.0]
        v1 = [1.0, 1.0, 0.0]
        v2 = [-1.0, 1.0, 0.0]
        v3 = [-1.0, -1.0, 0.0]

        vertices = np.array([v0, v1, v2, v3])
        # Definizione di tutte le facce (comprese quelle della Sezione 2)
        faces = np.array([
            # Sezione 1
            [0, 1, 2], [0, 2, 3], # Muro fisso 
        ])

        # Generazione della mesh finale
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    elif case == "canyon":
        path = meshes_dir / "simple_canyon.ply"
        mesh = trimesh.load_mesh(path)

    elif case == "florence":
        path = meshes_dir / "florence.ply"
        mesh = trimesh.load_mesh(path)

    elif case == "real_lunar":
        file = tifs_dir / "LDEM_875S_20M.tif"
        _, _, _, _, mesh = tif2mesh(file) 

    elif case == "triangoli":
        pitch_angle = np.radians(45)  # Inclinazione rispetto al piano XY
        spacing = 3.0                 # Distanza totale tra i centri lungo X
        offset_x = spacing / 2.0
        z_clearance = 2.0             # Distanza in Z di TX e RX sopra i centroidi

        # Vertici di base di un triangolo nel piano YZ (centrato nell'origine)
        # Base sull'asse Y, altezza lungo l'asse Z
        v0_local = np.array([0.0, -1.0, 0.0])
        v1_local = np.array([0.0,  1.0, 0.0])
        v2_local = np.array([0.0,  0.0, 2.0])

        # Funzione di utilità per la matrice di rotazione attorno all'asse Y
        def rot_y(theta):
            return np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])

        # --- Triangolo 1 (A destra, inclinato verso il centro) ---
        # Ruotiamo di +pitch_angle e trasliamo a +offset_x
        R1 = rot_y(pitch_angle)
        v0 = R1 @ v0_local + np.array([offset_x, 0.0, 0.0])
        v1 = R1 @ v1_local + np.array([offset_x, 0.0, 0.0])
        v2 = R1 @ v2_local + np.array([offset_x, 0.0, 0.0])

        # --- Triangolo 2 (A sinistra, inclinato verso il centro) ---
        # Ruotiamo di -pitch_angle e trasliamo a -offset_x
        R2 = rot_y(-pitch_angle)
        v3 = R2 @ v0_local + np.array([-offset_x, 0.0, 0.0])
        v4 = R2 @ v1_local + np.array([-offset_x, 0.0, 0.0])
        v5 = R2 @ v2_local + np.array([-offset_x, 0.0, 0.0])

        vertices = np.array([v0, v1, v2, v3, v4, v5])
        
        # Facce orientate in modo che le normali puntino verso lo spazio interno
        faces = np.array([
            [0, 2, 1],  # Triangolo 1 (A destra, guarda verso l'asse -X)
            [3, 4, 5]   # Triangolo 2 (A sinistra, guarda verso l'asse +X)
        ])

        # Generazione della mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh