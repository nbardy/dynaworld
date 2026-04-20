from camera import build_plucker_ray_grid, make_default_camera
from runtime_types import GaussianSequence

from .blocks import TokenGSBackbone


class TokenGS(TokenGSBackbone):
    def forward(self, img, camera=None):
        _, _, _, width = img.shape
        if camera is None:
            camera = make_default_camera(image_size=width, device=img.device)
        plucker_grid = build_plucker_ray_grid(camera, image_size=width, device=img.device, channels_first=True)
        xyz, scales, quats, opacities, rgbs = self.predict_gaussians(img, plucker_grid)
        return GaussianSequence(xyz=xyz, scales=scales, quats=quats, opacities=opacities, rgbs=rgbs)
