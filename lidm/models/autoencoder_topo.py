import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from scipy.spatial import Delaunay

# Import TopoLayer dependencies

from topologylayer.functional.utils_dionysus import top_batch_cost
from topologylayer.functional.levelset_dionysus import Diagramlayer as DiagramlayerToplevel
diagramlayerToplevel = DiagramlayerToplevel.apply


# Import Encoder and Decoder from model_topolidm
from ..modules.diffusion.model_topolidm import Encoder, Decoder


def setup_filtration(height, width):
    """
    Build Delaunay triangulation and initialize the filtration (simplicial complex)
    based on feature map dimensions, following GLiDR_kitti.py.
    """
    axis_x = np.arange(0, width)
    axis_y = np.arange(0, height)
    grid_axes = np.array(np.meshgrid(axis_x, axis_y))
    grid_axes = np.transpose(grid_axes, (1, 2, 0))
    
    tri = Delaunay(grid_axes.reshape([-1, 2]))
    faces = tri.simplices.copy()
    
    # Initialize filtration structure
    if DiagramlayerToplevel is not None:
        F_struct = DiagramlayerToplevel().init_filtration(faces)
        return F_struct
    return None


class TopoAutoencoder(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 image_key="image",
                 learning_rate=1e-4,
                 topo_weight=1.0,
                 ckpt_path=None,
                 **kwargs
                 ):
        super().__init__()
        self.image_key = image_key
        self.learning_rate = learning_rate
        self.topo_weight = topo_weight

        # 1. Instantiate Encoder and Decoder
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        # 2. Pre-build two filtrations on a 4x4 grid (matching GLiDR's 16-vertex Delaunay mesh,
        #    which greatly reduces persistent homology computation cost)
        topo_h, topo_w = 4, 4

        self.F_input = setup_filtration(topo_h, topo_w)
        self.F_latent = setup_filtration(topo_h, topo_w)

        # 3. Load pretrained weights if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        """Encode input to latent space, returning (B, C, H, W)."""
        z, _, _ = self.encoder(x)
        return z

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x):
        # Encoder returns z (B, C, H, W) and intermediate features L2, L4 (B, N, D) for topology loss
        z, L2_feat, L4_feat = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z, L2_feat, L4_feat

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[:, None]
        return x

    def _reduce_to_scalar_field(self, tensor, is_image=False):
        """
        Reduce a multi-channel feature map to a 16-point scalar field (matching the 4x4 filtration)
        for 0-dimensional topology lifetime computation via top_batch_cost.
        """
        topo_h, topo_w = 4, 4
        if is_image:
            # Image (B, C, 64, 1024) -> downsample to (B, C, 4, 16) -> (B, 16)
            B = tensor.shape[0]
            tensor = tensor.view(B, -1, topo_h, tensor.shape[2] // topo_h, topo_w, tensor.shape[3] // topo_w)
            # Average-pool each spatial block to get (B, C, 4, 16)
            tensor = tensor.mean(dim=3).mean(dim=3)  # remove the two intermediate block dimensions
            return tensor[:, 0, :, :].reshape(B, -1)  # (B, 64)
        else:
            # Point features (B, N, D) -> spatially sample down to 16 points -> (B, 16)
            B, N, D = tensor.shape
            tensor = tensor.view(B, topo_h, topo_w, N // (topo_h * topo_w), D)
            tensor = tensor.mean(dim=3)  # (B, 4, 16, D)
            return tensor.max(dim=-1)[0].reshape(B, -1)  # (B, 64)

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        
        # 1. Forward pass
        x_rec, z, L2_feat, L4_feat = self(inputs)

        # 2. L1 reconstruction loss
        loss_l1 = F.l1_loss(inputs, x_rec)

        # 3. Topological regularization loss
        loss_topo = torch.tensor(0.0, device=self.device)

        if self.F_input is not None and self.F_latent is not None:
            # Reduce tensors and move to CPU for topology loss (Dionysus requires CPU memory)
            # a. Topology constraint on reconstructed image \hat{I}
            x_rec_scalar = self._reduce_to_scalar_field(x_rec, is_image=True)
            topo_loss_out = top_batch_cost(x_rec_scalar.cpu(), diagramlayerToplevel, self.F_input)

            # b. Topology constraint on intermediate feature L2
            L2_scalar = self._reduce_to_scalar_field(L2_feat, is_image=False)
            topo_loss_L2 = top_batch_cost(L2_scalar.cpu(), diagramlayerToplevel, self.F_latent)

            # c. Topology constraint on intermediate feature L4
            L4_scalar = self._reduce_to_scalar_field(L4_feat, is_image=False)
            topo_loss_L4 = top_batch_cost(L4_scalar.cpu(), diagramlayerToplevel, self.F_latent)

            loss_topo = topo_loss_out + topo_loss_L2 + topo_loss_L4

        # 4. Total loss
        total_loss = loss_l1 + self.topo_weight * loss_topo

        # Logging
        self.log("train/loss_l1", loss_l1, prog_bar=True)
        self.log("train/loss_topo", loss_topo, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        x_rec, z, _, _ = self(inputs)
        val_loss = F.l1_loss(inputs, x_rec)
        
        self.log("val/rec_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        # Autoencoder optimizer only (no discriminator)
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        # Return a scheduler here if LR decay is needed
        return [opt_ae]

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, z, _, _ = self(x)
        
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log