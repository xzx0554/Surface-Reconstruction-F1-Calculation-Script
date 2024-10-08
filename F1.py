import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import csv

def read_ply(file_path):

    with open(file_path, 'r') as f:
        points = []
        inside_vertex_section = False
        vertex_count = 0
        
        for line in f:
            line = line.strip()  
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                inside_vertex_section = True
                continue
            
            if inside_vertex_section:
                if line.startswith('end_header'):
                    continue 
                if line.startswith('property') or line.startswith('comment'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3: 
                    try:
                        x, y, z = map(float, parts[:3])
                        points.append([x, y, z])
                    except ValueError:
                        continue 
    return np.array(points), vertex_count

def read_obj(file_path):

    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        raise ValueError(f"Mesh at {file_path} has no vertices.")
    return mesh

def sample_points(mesh, number_of_points=100000):

    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    return np.asarray(pcd.points)

def compute_f1_score(gt_points, recon_points, threshold=0.5):

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
    gt_kdtree = o3d.geometry.KDTreeFlann(gt_pcd)
    
    recon_pcd = o3d.geometry.PointCloud()
    recon_pcd.points = o3d.utility.Vector3dVector(recon_points)
    recon_kdtree = o3d.geometry.KDTreeFlann(recon_pcd)
    
    precisions = []
    for point in recon_points:
        [_, idx, dist] = gt_kdtree.search_knn_vector_3d(point, 1)
        if np.sqrt(dist[0]) < threshold:
            precisions.append(1)
        else:
            precisions.append(0)
    precision = np.sum(precisions) / len(precisions)
    
    recalls = []
    for point in gt_points:
        [_, idx, dist] = recon_kdtree.search_knn_vector_3d(point, 1)
        if np.sqrt(dist[0]) < threshold:
            recalls.append(1)
        else:
            recalls.append(0)
    recall = np.sum(recalls) / len(recalls)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1

def main():
    train_folder = './lingkeyuan30'
    ck_folder = './lingkeyuan30_check'
    output_file = 'lky_30_F1.csv'
    results = []
    train_files = [f for f in os.listdir(train_folder) if f.endswith('.ply')]
    
    for ply_file in tqdm(train_files):
        base_name = os.path.splitext(ply_file)[0]
        obj_folder = os.path.join(ck_folder, base_name )
        ply_file_path = os.path.join(train_folder, ply_file)
        for i in range(100,10100,100):
            obj_file = os.path.join(obj_folder, f'recon_iter_{i}.obj' )
            if not os.path.exists(obj_file):
                print(f"对应的OBJ文件未找到: {obj_file}")
                continue
            
            gt_points, gt_count = read_ply(ply_file_path)
            
            mesh = read_obj(obj_file)
            
            recon_points = sample_points(mesh, number_of_points=100000)
            
            precision, recall, f1 = compute_f1_score(gt_points, recon_points, threshold=0.1)
            results.append({
                    'Reference File': base_name,
                    'Recon_iter': i,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                })
            print(f"文件: {base_name}")
            print(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}\n")
    fieldnames = ['Reference', 'Recon_iter', 'Precision', 'Recall', 'Score']
    
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"所有结果已保存到 {output_file}")
    except Exception as e:
        print(f"保存到 CSV 时出错: {e}")
if __name__ == "__main__":
    main()
