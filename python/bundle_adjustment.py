import numpy as np
import g2o

class MonoBA(g2o.SparseOptimizer):
    def __init__(self):
        super().__ini__()

        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        terminate = g2o.SparseOptimizerTerminateAction()
        terminate.set_gain_threshold(1e-6)
        super().add_post_iteration_action(terminate)

        self.delta = np.sqrt(5.991)
        self.aborted = False

    def set_cam(self, focal_length, principal_point):
        cam = g2o.CameraParameters(focal_length, principal_point, 0)
        cam.set_id(0)
        super().add_parameter(cam)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)
        try:
            return not self.aborted
        finally:
            self.aborted = False

    def add_pose(self, pose_id, pose, cam, fixed=False):
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(pose_id * 2)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_marginalized(marginalized)
        v_p.set_estimate(point)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, id, point_id, pose_id, meas, info=np.identity(3)):
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(meas)
        edge.set_information(info)
        kernel = g2o.RobustKernelHuber()
        edge.set_robust_kernel(kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id * 2).estimate()

    def get_point(self, id):
        return self.vertex(id * 2).estimate()

    def abort(self):
        self.aborted = True
        
