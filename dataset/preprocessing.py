import torch
import trimesh


class MoveToOriginTransform(object):
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh.vertices -= mesh.vertices.mean(axis=0)
        return mesh


class PreTransform(object):
    def __init__(self, classes=17):
        self.classes = classes

    def __call__(self, data):
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = data
        mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(mesh_triangles.cpu().detach().numpy()))

        points = torch.from_numpy(mesh.vertices)
        # v_normals = torch.from_numpy(mesh.vertex_normals)
        v_normals = mesh.vertex_normals

        s, _ = mesh_faces.size()
        x = torch.zeros(s, 24).float()
        x[:, :3] = mesh_triangles[:, 0]
        x[:, 3:6] = mesh_triangles[:, 1]
        x[:, 6:9] = mesh_triangles[:, 2]
        x[:, 9:12] = mesh_triangles.mean(dim=1)
        x[:, 12:15] = mesh_vertices_normals[:, 0]
        x[:, 15:18] = mesh_vertices_normals[:, 1]
        x[:, 18:21] = mesh_vertices_normals[:, 2]
        x[:, 21:] = mesh_face_normals

        maxs = points.max(dim=0)[0]
        mins = points.min(dim=0)[0]
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = v_normals.mean(axis=0)
        nstds = v_normals.std(axis=0)
        nmeans_f = mesh_face_normals.mean(axis=0)
        nstds_f = mesh_face_normals.std(axis=0)
        for i in range(3):
            # normalize coordinate
            x[:, i] = (x[:, i] - means[i]) / stds[i]  # point 1
            x[:, i + 3] = (x[:, i + 3] - means[i]) / stds[i]  # point 2
            x[:, i + 6] = (x[:, i + 6] - means[i]) / stds[i]  # point 3
            x[:, i + 9] = (x[:, i + 9] - mins[i]) / (maxs[i] - mins[i])  # centre
            # normalize normal vector
            x[:, i + 12] = (x[:, i + 12] - nmeans[i]) / nstds[i]  # normal1
            x[:, i + 15] = (x[:, i + 15] - nmeans[i]) / nstds[i]  # normal2
            x[:, i + 18] = (x[:, i + 18] - nmeans[i]) / nstds[i]  # normal3
            x[:, i + 21] = (x[:, i + 21] - nmeans_f[i]) / nstds_f[i]  # face normal

        pos = x[:, 9:12]

        return pos, x, labels


class PreTransformDBGANet(object):
    def __init__(self, classes=17):
        self.classes = classes

    def __call__(self, data):
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = data
        mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(mesh_triangles.cpu().detach().numpy()))
        points = torch.from_numpy(mesh.vertices)
        # v_normals = torch.from_numpy(mesh.vertex_normals)
        v_normals = mesh.vertex_normals

        s, _ = mesh_faces.size()
        x = torch.zeros(s, 6).float()
        x[:, 0:3] = mesh_triangles.mean(dim=1)
        x[:, 3:] = mesh_face_normals

        x[:, 0:3], _, _ = self.pc_normalize(x[:, 0:3])
        x[:, 3:6], _, _ = self.pc_normalize(x[:, 3:6])

        pos = x[:, :3]
        lab_tooth_dict = {}
        for i in range(self.classes):
            lab_tooth_dict[i] = []
        for i, lab in enumerate(labels.reshape(-1)):
            lab_tooth_dict[lab.item()].append(list(pos[i]))

        barycenter = torch.zeros([self.classes, 3])
        for k, v in lab_tooth_dict.items():
            if v == []:
                continue
            temp = torch.tensor(lab_tooth_dict[k])
            barycenter[k] = temp.mean(axis=0)
        barycenter_label = torch.zeros([self.classes, ])
        for i, j in enumerate(barycenter_label):
            barycenter_label[i] = 1
            if barycenter[i][0] == 0 and barycenter[i][1] == 0 and barycenter[i][2] == 0:
                barycenter_label[i] = 0
        barycenter_label = barycenter_label[1:]
        barycenter = barycenter[1:]
        barycenter_label = barycenter_label.reshape(-1, 1).transpose(1, 0)

        return x, labels, barycenter, barycenter_label

    @staticmethod
    def pc_normalize(pc):
        centroid = torch.mean(pc, dim=0)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
        pc = pc / m
        return pc, centroid, m


class PreTransformTHISNet(object):
    def __init__(self, classes=17):
        self.classes = classes

    def __call__(self, data):
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = data
        mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(mesh_triangles.cpu().detach().numpy()))

        points = torch.from_numpy(mesh.vertices)
        # v_normals = torch.from_numpy(mesh.vertex_normals)
        # v_normals = mesh.vertex_normals

        s, _ = mesh_faces.size()
        x = torch.zeros(s, 24).float()
        x[:, :3] = mesh_triangles.mean(dim=1)
        x[:, 3:6] = mesh_triangles[:, 0]
        x[:, 6:9] = mesh_triangles[:, 1]
        x[:, 9:12] = mesh_triangles[:, 2]
        x[:, 12:15] = mesh_face_normals
        x[:, 15:18] = mesh_vertices_normals[:, 0]
        x[:, 18:21] = mesh_vertices_normals[:, 1]
        x[:, 21:] = mesh_vertices_normals[:, 2]

        means = points.mean(axis=0)
        points -= means
        max_len = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
        for i in range(3):
            # normalize coordinate
            x[:, i] = (x[:, i] - means[i]) / max_len.item()  # point 1
            x[:, i + 3] = (x[:, i + 3] - means[i]) / max_len.item()   # point 2
            x[:, i + 6] = (x[:, i + 6] - means[i]) / max_len.item()   # point 3
            x[:, i + 9] = (x[:, i + 9] - means[i]) / max_len.item()   # centre

        return x, labels


class PreTransformMeshseg(object):
    def __init__(self, classes=17):
        self.classes = classes

    def __call__(self, data):
        mesh_faces, mesh_triangles, mesh_vertices_normals, mesh_face_normals, labels = data
        mesh = trimesh.Trimesh(**trimesh.triangles.to_kwargs(mesh_triangles.cpu().detach().numpy()))

        points = torch.from_numpy(mesh.vertices)
        # v_normals = torch.from_numpy(mesh.vertex_normals)
        v_normals = mesh.vertex_normals

        s, _ = mesh_faces.size()
        x = torch.zeros(s, 15).float()
        x[:, :3] = mesh_triangles[:, 0]
        x[:, 3:6] = mesh_triangles[:, 1]
        x[:, 6:9] = mesh_triangles[:, 2]
        x[:, 9:12] = mesh_triangles.mean(dim=1)
        x[:, 12:] = mesh_face_normals

        maxs = points.max(dim=0)[0]
        mins = points.min(dim=0)[0]
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans_f = mesh_face_normals.mean(axis=0)
        nstds_f = mesh_face_normals.std(axis=0)
        for i in range(3):
            # normalize coordinate
            x[:, i] = (x[:, i] - means[i]) / stds[i]  # point 1
            x[:, i + 3] = (x[:, i + 3] - means[i]) / stds[i]  # point 2
            x[:, i + 6] = (x[:, i + 6] - means[i]) / stds[i]  # point 3
            x[:, i + 9] = (x[:, i + 9] - mins[i]) / (maxs[i] - mins[i])  # centre
            # normalize normal vector
            x[:, i + 12] = (x[:, i + 12] - nmeans_f[i]) / nstds_f[i]  # face normal

        pos = x[:, 9:12]
        a_s = torch.zeros((s, s))
        a_l = torch.zeros((s, s))
        D = torch.cdist(pos, pos)
        a_s[D < 0.1] = 1.0
        a_s = a_s / torch.mm(torch.sum(a_s, dim=1, keepdims=True), torch.ones((1, s)))
        a_l[D < 0.2] = 1.0
        a_l = a_l / torch.mm(torch.sum(a_l, dim=1, keepdims=True), torch.ones((1, s)))

        return a_s, a_l, x, labels

