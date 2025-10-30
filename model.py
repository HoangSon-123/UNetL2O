# file: model.py
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

inference = torch.Tensor
input_data = torch.Tensor
dual = torch.Tensor
# -------------------------------------------------
# 1. BASE MODEL: ImplicitL2OModel
# -------------------------------------------------
class ImplicitL2OModel(ABC, nn.Module):
    def device(self):
        return next(self.parameters()).data.device

    def assign_cert_model(self, cert_model):
        self._cert_model = cert_model

    def get_certs(self, x: inference, d: input_data): # pyright: ignore[reportInvalidTypeForm]
        valid_cert_model = self._cert_model is not None
        assert valid_cert_model, 'Certificate model must be assigned first'
        return self._cert_model.get_certs(x, d)

    @abstractmethod
    def _apply_T(self):
        pass

    @abstractmethod
    def _get_conv_crit(self):
        pass

    @abstractmethod
    def forward(self):
        pass

# -------------------------------------------------
# 2. MODEL: CT_L2O_Model
# -------------------------------------------------
class CT_L2O_Model(ImplicitL2OModel):
    def __init__(self,
                 A,
                 lambd=0.1,
                 alpha=0.1,
                 beta=0.1,
                 delta=-5.0,
                 K_out_channels=2,
                 max_depth=200):
        super().__init__()
        self.A = A
        self.At = A.t()
        self.max_depth = max_depth
        self.K_out_channels = K_out_channels
        self.fixed_point_error = 0.0
        self.fidelity_rel_norm_error = 0.0

        # trainable parameters
        self.delta = nn.Parameter(delta * torch.ones(1, device=A.device))
        self.alpha = nn.Parameter(alpha*torch.ones(1, device=A.device))
        self.lambd = nn.Parameter(lambd*torch.ones(1, device=A.device))
        self.beta = nn.Parameter(beta*torch.ones(1, device=A.device))
        self.leaky_relu = nn.LeakyReLU(0.1)

        # layers for R
        self.conv1 = nn.Conv2d(in_channels=K_out_channels,
                               out_channels=K_out_channels,
                               kernel_size=5,
                               stride=1,
                               padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=K_out_channels,
                               out_channels=K_out_channels,
                               kernel_size=5,
                               stride=1,
                               padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=K_out_channels,
                               out_channels=K_out_channels,
                               kernel_size=5,
                               stride=1,
                               padding=(2, 2))
        # layers for K
        self.convK = nn.Conv2d(in_channels=1,
                               out_channels=K_out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.convK_T = nn.ConvTranspose2d(in_channels=K_out_channels,
                                          out_channels=1,
                                          kernel_size=3,
                                          padding=1,
                                          bias=False)

    def _get_conv_crit(self, x, x_prev, d, tol=1.0e-2):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        d = d.view(batch_size, -1)
        x_prev = x_prev.view(batch_size, -1)

        res_norm = torch.max(torch.norm(x - x_prev, dim=1))
        residual_conv = res_norm <= tol
        return residual_conv

    def name(self) -> str:
        return "CTModel" # (Có thể đổi tên thành "CT_L2O_Model")

    # (Lưu ý: hàm device() đã có trong class cha ImplicitL2OModel)
    # def device(self):
    #     return next(self.parameters()).data.device

    def box_proj(self, x):
        return torch.clamp(x, min=0.0, max=1.0)

    def K(self, x):
        batch_size = x.shape[1]
        x = x.permute(1, 0).view(batch_size, 1, 128, 128)
        x = self.convK(x)
        x = x.view(batch_size, -1).permute(1, 0)
        return x

    def Kt(self, p):
        batch_size = p.shape[1]
        p = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)
        p = self.convK_T(p)
        p = p.view(batch_size, -1).permute(1, 0)
        return p

    def ball_proj(self, w, d, delta, proj_weight=0.99):
        delta = torch.exp(self.delta) # Sử dụng self.delta
        dist = torch.norm(w - d, dim=0)
        d_norm = torch.norm(d, dim=0)
        dist[dist <= 1e-10] = 1e-10
        scale = torch.minimum(torch.ones(dist.shape, device=d.device),
                              delta*d_norm/dist)
        proj = d + scale * (w - d)
        other = d + delta * d_norm * (w - d) / dist

        if self.training:
            return proj_weight * proj + (1.0 - proj_weight) * other
        else:
            return proj

    def R(self, p):
        batch_size = p.shape[1]
        p_res = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)
        
        # Sửa lại theo code bạn cung cấp (dùng residual cho mỗi lớp)
        p_res = p_res + self.leaky_relu(self.conv1(p_res))
        p_res = p_res + self.leaky_relu(self.conv2(p_res))
        p_res = p_res + self.leaky_relu(self.conv3(p_res))

        p_res = p_res.view(batch_size, -1).permute(1, 0)
        p_res = p_res.view(self.K_out_channels*(128**2), batch_size)
        return p_res # Trả về p_res (đã bao gồm p gốc)

    def _apply_T(self, x: inference, d: input_data, return_tuple=False): # type: ignore
        batch_size = x.shape[0]

        d = d.view(d.shape[0], -1).to(self.device())
        d = d.permute(1, 0)
        xk = x.view(x.shape[0], -1)
        xk = xk.permute(1, 0)
        pk = self.K(xk)
        wk = torch.matmul(self.A, xk)
        nuk1 = torch.zeros(pk.size(), device=self.device())
        nuk2 = torch.zeros(d.size(), device=self.device())

        alpha = torch.clamp(self.alpha.data, min=0, max=2)
        beta = torch.clamp(self.beta.data, min=0, max=2)
        lambd = torch.clamp(self.lambd.data, min=0, max=2)
        delta = self.delta.data # Sử dụng self.delta

        # pk step
        pk = pk + lambd*(nuk1 + alpha * (self.K(xk) - pk))
        pk = self.R(pk)

        # wk step
        Axk = torch.matmul(self.A, xk)
        res_temp = nuk2 + alpha * (Axk - wk)
        temp_term = wk + lambd * res_temp
        wk = self.ball_proj(temp_term, d, delta) # Truyền delta vào

        # nuk1 step
        res_temp = self.K(xk) - pk
        nuk1_plus = nuk1 + alpha * res_temp

        # nuk2 step
        res_temp = Axk - wk
        nuk2_plus = nuk2 + alpha * res_temp

        # rk step
        self.convK_T.weight.data = self.convK.weight.data
        rk = self.Kt(2*nuk1_plus - nuk1)
        rk = rk + torch.matmul(self.At, 2*nuk2_plus - nuk2)

        # xk step
        xk = torch.clamp(xk - beta * rk, min=0, max=1)
        
        xk_res = xk.permute(1, 0).view(batch_size, 1, 128, 128)

        if return_tuple:
            return xk_res, nuk1_plus, pk
        else:
            return xk_res

    def forward(self, d, depth_warning=False, return_depth=False, tol=1e-3, return_all_vars=False):
        with torch.no_grad():
            self.depth = 0.0
            x = torch.zeros((d.size()[0], 1, 128, 128),
                            device=self.device())
            x_prev = np.inf*torch.ones(x.shape, device=self.device())
            all_samp_conv = False

            while not all_samp_conv and self.depth < self.max_depth:
                x_prev = x.clone()
                x = self._apply_T(x, d)
                all_samp_conv = self._get_conv_crit(x,
                                                    x_prev,
                                                    d,
                                                    tol=tol)
                self.depth += 1

        if self.depth >= self.max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        self.fixed_point_error = torch.max(torch.norm(x - x_prev, dim=1))

        Tx, nuk1, pk = self._apply_T(x, d, return_tuple=True)
        
        if return_depth:
            return Tx, self.depth
        elif return_all_vars:
            return Tx, nuk1, pk
        else:
            return Tx

# -------------------------------------------------
# 3. MODEL: New_CT_L2O_Model 
# -------------------------------------------------
import torch
import torch.nn as nn
import numpy as np


class UNetL2O(ImplicitL2OModel):
    """
    Learned optimization model for CT image reconstruction using an unrolled
    ADMM-like iterative structure. The regularizer is implemented as a small
    U-Net–like CNN acting as a learned proximal operator.

    Args:
        A (torch.Tensor): Forward projection matrix.
        lambd (float): Regularization parameter.
        alpha (float): Step size for the dual variable.
        beta (float): Step size for the primal variable.
        delta (float): Log-space scaling parameter for projection.
        K_out_channels (int): Number of feature channels in the learned operator K.
        max_depth (int): Maximum number of unrolled iterations.
    """

    def __init__(self,
                 A,
                 lambd=0.1,
                 alpha=0.1,
                 beta=0.1,
                 delta=-5.0,
                 K_out_channels=2,
                 max_depth=200):
        super().__init__()
        self.A = A
        self.At = A.t()
        self.max_depth = max_depth
        self.K_out_channels = K_out_channels
        self.fixed_point_error = 0.0
        self.fidelity_rel_norm_error = 0.0

        # Trainable parameters
        self.delta = nn.Parameter(delta * torch.ones(1, device=A.device))
        self.alpha = nn.Parameter(alpha * torch.ones(1, device=A.device))
        self.lambd = nn.Parameter(lambd * torch.ones(1, device=A.device))
        self.beta = nn.Parameter(beta * torch.ones(1, device=A.device))
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Convolutional operators K and K^T
        self.convK = nn.Conv2d(in_channels=1,
                               out_channels=K_out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.convK_T = nn.ConvTranspose2d(in_channels=K_out_channels,
                                          out_channels=1,
                                          kernel_size=3,
                                          padding=1,
                                          bias=False)

        # Learned proximal operator R (U-Net–like architecture)
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(K_out_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck with dilated convolutions
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )

        # Decoder with skip connections
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, K_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(K_out_channels),
            nn.LeakyReLU(0.1)
        )

    def name(self) -> str:
        return "UNetL2O"

    def box_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp the image intensity values to [0, 1]."""
        return torch.clamp(x, min=0.0, max=1.0)

    def _get_conv_crit(self, x, x_prev, d, tol=1.0e-2):
        """Check convergence criterion based on relative change in x."""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x_prev = x_prev.view(batch_size, -1)
        res_norm = torch.max(torch.norm(x - x_prev, dim=1))
        return res_norm <= tol

    def K(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned linear operator K."""
        batch_size = x.shape[1]
        x = x.permute(1, 0).view(batch_size, 1, 128, 128)
        x = self.convK(x)
        return x.view(batch_size, -1).permute(1, 0)

    def Kt(self, p: torch.Tensor) -> torch.Tensor:
        """Apply the transpose operator Kᵗ."""
        batch_size = p.shape[1]
        p = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)
        p = self.convK_T(p)
        return p.view(batch_size, -1).permute(1, 0)

    def ball_proj(self, w, d, delta, proj_weight=0.99):
        """
        Projection of w onto a ball around d with adaptive radius exp(delta)*||d||.
        """
        delta_val = torch.exp(self.delta)
        dist = torch.norm(w - d, dim=0)
        d_norm = torch.norm(d, dim=0)
        dist[dist <= 1e-10] = 1e-10

        scale = torch.minimum(torch.ones_like(dist),
                              delta_val * d_norm / dist)
        proj = d + scale * (w - d)
        other = d + delta_val * d_norm * (w - d) / dist

        if self.training:
            return proj_weight * proj + (1.0 - proj_weight) * other
        else:
            return proj

    def R(self, p: torch.Tensor) -> torch.Tensor:
        """Learned proximal operator implemented as a U-Net block."""
        batch_size = p.shape[1]
        p_res = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)

        # Encoder
        enc1 = self.enc1(p_res)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        # Bottleneck
        bottle = self.bottleneck(pool2)

        # Decoder with skip connections
        up2 = self.upconv2(bottle)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        # Residual connection
        p_res = p_res + dec1
        p_res = p_res.view(batch_size, -1).permute(1, 0)
        return p_res

    def _apply_T(self, x, d, return_tuple=False):
        """One unrolled iteration step of the learned optimization."""
        batch_size = x.shape[0]

        d_flat = d.view(d.shape[0], -1).to(self.device()).permute(1, 0)
        xk = x.view(batch_size, -1).permute(1, 0)
        pk = self.K(xk)
        wk = torch.matmul(self.A, xk)
        nuk1 = torch.zeros_like(pk)
        nuk2 = torch.zeros_like(d_flat)

        alpha = torch.clamp(self.alpha.data, min=0, max=2)
        beta = torch.clamp(self.beta.data, min=0, max=2)
        lambd = torch.clamp(self.lambd.data, min=0, max=2)
        delta = self.delta.data

        # pk step
        pk = pk + lambd * (nuk1 + alpha * (self.K(xk) - pk))
        pk = self.R(pk)

        # wk step
        Axk = torch.matmul(self.A, xk)
        wk = self.ball_proj(wk + lambd * (nuk2 + alpha * (Axk - wk)), d_flat, delta)

        # Dual updates
        nuk1_plus = nuk1 + alpha * (self.K(xk) - pk)
        nuk2_plus = nuk2 + alpha * (Axk - wk)

        # rk step
        self.convK_T.weight.data = self.convK.weight.data
        rk = self.Kt(2 * nuk1_plus - nuk1) + torch.matmul(self.At, 2 * nuk2_plus - nuk2)

        # xk update
        xk = torch.clamp(xk - beta * rk, min=0, max=1)
        xk_res = xk.permute(1, 0).view(batch_size, 1, 128, 128)

        if return_tuple:
            return xk_res, nuk1_plus, pk
        return xk_res

    def forward(self, d, depth_warning=False, return_depth=False, tol=1e-3, return_all_vars=False):
        """Forward pass performing iterative reconstruction."""
        with torch.no_grad():
            self.depth = 0
            x = torch.zeros((d.size(0), 1, 128, 128), device=self.device())
            x_prev = np.inf * torch.ones_like(x)
            all_conv = False

            while not all_conv and self.depth < self.max_depth:
                x_prev = x.clone()
                x = self._apply_T(x, d)
                all_conv = self._get_conv_crit(x, x_prev, d, tol=tol)
                self.depth += 1

        if self.depth >= self.max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Breaking Forward Loop\n")

        self.fixed_point_error = torch.max(torch.norm(x - x_prev, dim=1))

        # Final iteration (with gradient tracking)
        Tx, nuk1, pk = self._apply_T(x, d, return_tuple=True)

        if return_depth:
            return Tx, self.depth
        elif return_all_vars:
            return Tx, nuk1, pk
        else:
            return Tx

    ''' Model (U-Net) để tái tạo ảnh CT. '''
    def __init__(self,
                 A,
                 lambd=0.1,
                 alpha=0.1,
                 beta=0.1,
                 delta=-5.0,
                 K_out_channels=2,
                 max_depth=200):
        super().__init__()
        self.A = A
        self.At = A.t()
        self.max_depth = max_depth
        self.K_out_channels = K_out_channels
        self.fixed_point_error = 0.0
        self.fidelity_rel_norm_error = 0.0

        # trainable parameters
        self.delta = nn.Parameter(delta * torch.ones(1, device=A.device))
        self.alpha = nn.Parameter(alpha * torch.ones(1, device=A.device))
        self.lambd = nn.Parameter(lambd * torch.ones(1, device=A.device))
        self.beta = nn.Parameter(beta * torch.ones(1, device=A.device))
        self.leaky_relu = nn.LeakyReLU(0.1)

        # layers for K
        self.convK = nn.Conv2d(in_channels=1,
                               out_channels=K_out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.convK_T = nn.ConvTranspose2d(in_channels=K_out_channels,
                                          out_channels=1,
                                          kernel_size=3,
                                          padding=1,
                                          bias=False)

        # layers for R (U-Net inspired architecture)
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(K_out_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck with dilated convolutions
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, K_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(K_out_channels),
            nn.LeakyReLU(0.1)
        )

    def _get_conv_crit(self, x, x_prev, d, tol=1.0e-2):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        d = d.view(batch_size, -1)
        x_prev = x_prev.view(batch_size, -1)

        res_norm = torch.max(torch.norm(x - x_prev, dim=1))
        residual_conv = res_norm <= tol
        return residual_conv

    def name(self) -> str:
        return "New_CT_L2O_Model"

    def box_proj(self, x):
        return torch.clamp(x, min=0.0, max=1.0)

    def K(self, x):
        batch_size = x.shape[1]
        x = x.permute(1, 0).view(batch_size, 1, 128, 128)
        x = self.convK(x)
        x = x.view(batch_size, -1).permute(1, 0)
        return x

    def Kt(self, p):
        batch_size = p.shape[1]
        p = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)
        p = self.convK_T(p)
        p = p.view(batch_size, -1).permute(1, 0)
        return p

    def ball_proj(self, w, d, delta, proj_weight=0.99):
        # Chuyển self.delta sang tensor trước khi exp
        delta_val = torch.exp(self.delta) 

        dist = torch.norm(w - d, dim=0)
        d_norm = torch.norm(d, dim=0)
        dist[dist <= 1e-10] = 1e-10
        scale = torch.minimum(torch.ones(dist.shape, device=d.device),
                              delta_val * d_norm / dist)
        proj = d + scale * (w - d)
        other = d + delta_val * d_norm * (w - d) / dist

        if self.training:
            return proj_weight * proj + (1.0 - proj_weight) * other
        else:
            return proj

    def R(self, p):
        batch_size = p.shape[1]
        p_res = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)

        # Encoder
        enc1 = self.enc1(p_res)
        pool1 = self.pool1(enc1)
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        # Bottleneck
        bottle = self.bottleneck(pool2)
        
        # Decoder with skip connections
        up2 = self.upconv2(bottle)
        dec2_input = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(dec2_input)
        
        up1 = self.upconv1(dec2)
        dec1_input = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(dec1_input)
        
        # Residual connection
        p_res = p_res + dec1
        
        p_res = p_res.view(batch_size, -1).permute(1, 0)
        p_res = p_res.view(self.K_out_channels * (128**2), batch_size)

        return p_res # Trả về p + p_res (hoặc chỉ p_res tùy theo logic, notebook của bạn là p + dec1)

    def _apply_T(self, x: inference, d: input_data, return_tuple=False): # type: ignore
        batch_size = x.shape[0]

        d_flat = d.view(d.shape[0], -1).to(self.device())
        d_flat = d_flat.permute(1, 0)
        xk = x.view(x.shape[0], -1)
        xk = xk.permute(1, 0)
        pk = self.K(xk)
        wk = torch.matmul(self.A, xk)
        nuk1 = torch.zeros(pk.size(), device=self.device())
        nuk2 = torch.zeros(d_flat.size(), device=self.device())

        alpha = torch.clamp(self.alpha.data, min=0, max=2)
        beta = torch.clamp(self.beta.data, min=0, max=2)
        lambd = torch.clamp(self.lambd.data, min=0, max=2)
        delta = self.delta.data

        # pk step
        pk = pk + lambd*(nuk1 + alpha * (self.K(xk) - pk))
        pk = self.R(pk)

        # wk step
        Axk = torch.matmul(self.A, xk)
        res_temp = nuk2 + alpha * (Axk - wk)
        temp_term = wk + lambd * res_temp

        wk = self.ball_proj(temp_term, d_flat, delta)

        # nuk1 step
        res_temp = self.K(xk) - pk
        nuk1_plus = nuk1 + alpha * res_temp

        # nuk2 step
        res_temp = Axk - wk
        nuk2_plus = nuk2 + alpha * res_temp

        # rk step
        self.convK_T.weight.data = self.convK.weight.data
        rk = self.Kt(2*nuk1_plus - nuk1)
        rk = rk + torch.matmul(self.At, 2*nuk2_plus - nuk2)

        # xk step
        xk = torch.clamp(xk - beta * rk, min=0, max=1)
        
        xk_res = xk.permute(1, 0).view(batch_size, 1, 128, 128)
        
        if return_tuple:
            return xk_res, nuk1_plus, pk
        else:
            return xk_res

    def forward(self, d, depth_warning=False, return_depth=False, tol=1e-3, return_all_vars=False):
        with torch.no_grad():
            self.depth = 0.0
            x = torch.zeros((d.size()[0], 1, 128, 128),
                            device=self.device())
            x_prev = np.inf*torch.ones(x.shape, device=self.device())
            all_samp_conv = False

            while not all_samp_conv and self.depth < self.max_depth:
                x_prev = x.clone()
                x = self._apply_T(x, d)
                all_samp_conv = self._get_conv_crit(x,
                                                    x_prev,
                                                    d,
                                                    tol=tol)
                self.depth += 1

        if self.depth >= self.max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        self.fixed_point_error = torch.max(torch.norm(x - x_prev, dim=1))

        # Apply T *with* gradient tracking
        Tx, nuk1, pk = self._apply_T(x, d, return_tuple=True)
        
        if return_depth:
            return Tx, self.depth
        elif return_all_vars:
            return Tx, nuk1, pk
        else:
            return Tx
# -------------------------------------------------
# 4. MODEL: CT_FFPN_Model 
# -------------------------------------------------
class CT_FFPN_Model(nn.Module):
    def __init__(self, D, M, res_net_contraction=0.99):
        super().__init__()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.gamma = res_net_contraction
        self.num_channels = 44     
        self.D = D 
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_channels, 
                                              kernel_size=5, stride=1, 
                                              padding=(2,2)),
                                    nn.Conv2d(self.num_channels, 
                                              self.num_channels, kernel_size=5, 
                                              stride=1, padding=(2,2)),
                                    nn.Conv2d(self.num_channels, 
                                              self.num_channels, kernel_size=5, 
                                              stride=1, padding=(2,2)),
                                    nn.Conv2d(self.num_channels, 1, 
                                              kernel_size=5, stride=1, 
                                              padding=(2,2))])
        self.M = M
        self.Mt = M.t()         

    def name(self) -> str:
        return "CT_FFPN_Model"

    def device(self):
        return next(self.parameters()).data.device

    def _T(self, u, d):
        batch_size = u.shape[0]

        # Learned Gradient
        for idx, conv in enumerate(self.convs):
            u_ref = u if idx + 1 < len(self.convs) else u[:,0,:,:].view(batch_size,1,128,128)
            u = u_ref + self.leaky_relu(conv(u))
        u = torch.clamp(u, min=0, max=1.0e1)

        # Constraints Projection
        u_vec = u.view(batch_size, -1).to(self.device())
        u_vec = u_vec.permute(1,0).to(self.device())   
        d = d.view(batch_size,-1).to(self.device())
        d = d.permute(1,0)
        res = torch.matmul(self.Mt, self.M.matmul(u_vec) - d)
        res = 1.99 * torch.matmul(self.D.to(self.device()), res)
        res = res.permute(1,0)
        res = res.view(batch_size, 1, 128, 128).to(self.device())
        return u - res

    def normalize_lip_const(self, u, d):
        ''' Scale convolutions in R to make it gamma Lipschitz

            It should hold that |R(u,v) - R(w,v)| <= gamma * |u-w| for all u
            and w. If this doesn't hold, then we must rescale the convolution.
            Consider R = I + Conv. To rescale, ideally we multiply R by

                norm_fact = gamma * |u-w| / |R(u,v) - R(w,v)|,
            
            averaged over a batch of samples, i.e. R <-- norm_fact * R. The 
            issue is that ResNets include an identity operation, which we don't 
            wish to rescale. So, instead we use
                
                R <-- I + norm_fact * Conv,
            
            which is accurate up to an identity term scaled by (norm_fact - 1).
            If we do this often enough, then norm_fact ~ 1.0 and the identity 
            term is negligible.

            Note: BatchNorm and ReLUs are nonexpansive when...???
        '''
        noise_u = torch.randn(u.size(), device=self.device()) 
        w = u.clone() + noise_u
        w = w.to(self.device())
        Twd = self._T(w, d)
        Tud = self._T(u, d)
        T_diff_norm = torch.mean(torch.norm(Twd - Tud, dim=1))
        u_diff_norm = torch.mean(torch.norm(w - u, dim=1))
        R_is_gamma_lip = T_diff_norm <= self.gamma * u_diff_norm
        if not R_is_gamma_lip:
            normalize_factor = (self.gamma * u_diff_norm / T_diff_norm) ** (1.0 / len(self.convs))
            for i in range(len(self.convs)):
                self.convs[i].weight.data *= normalize_factor
                self.convs[i].bias.data *= normalize_factor

    def forward(self, d, eps=1.0e-3, max_depth=100, 
                depth_warning=False):
        ''' FPN forward prop

            With gradients detached, find fixed point. During forward iteration,
            u is updated via R(u,Q(d)) and Lipschitz constant estimates are
            refined. Gradient are attached performing one final step.
        '''         
        with torch.no_grad():
            self.depth = 0.0
            u = torch.zeros((d.size()[0], 1, 128, 128), 
                            device=self.device())
            u_prev = np.inf*torch.ones(u.shape, device=self.device())            
            all_samp_conv = False
            while not all_samp_conv and self.depth < max_depth:
                u_prev = u.clone()   
                u = self._T(u, d)
                res_norm = torch.max(torch.norm(u - u_prev, dim=1)) 
                # print('res_norm = ', res_norm)
                self.depth += 1.0
                all_samp_conv = res_norm <= eps
            
            if self.training:
                self.normalize_lip_const(u, d)

        if self.depth >= max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        return self._T(u, d) 

# -------------------------------------------------
# 5. MODEL: CT_UNet_Model 
# -------------------------------------------------
class CT_UNet_Model(nn.Module):
    """
    U-Net architecture for CT image reconstruction or enhancement.

    The model consists of three downsampling (contracting) blocks 
    followed by three upsampling (expanding) blocks. 
    Skip connections are used between corresponding encoder and decoder layers.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Encoder (contracting path)
        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        # Decoder (expanding path)
        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64 * 2, 32, 3, 1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Reconstructed or enhanced image tensor.
        """
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Decoder with skip connections
        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], dim=1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], dim=1))

        return upconv1

    def contract_block(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
        """
        Creates a contracting block with two convolutional layers, 
        followed by BatchNorm, ReLU, and a MaxPooling layer.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return block

    def expand_block(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> nn.Sequential:
        """
        Creates an expanding block with two convolutional layers 
        followed by BatchNorm, ReLU, and a transposed convolution for upsampling.
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return block

# -------------------------------------------------
# 6. MODEL: CT_TVM_Model 
# -------------------------------------------------
class CT_TVM_Model(nn.Module):
    def __init__(self,
                 A, lambd=0.1,
                 alpha=0.1,
                 beta=0.1,
                 eps=1e-1):
        super().__init__()
        self.A = A
        self.At = A.t()
        self.lambd = lambd
        self.alpha = alpha
        self.beta = beta
        self.shrink = torch.nn.Softshrink(lambd=lambd)
        self.eps = eps
        self.model_device = 'cpu'

    def name(self) -> str:
        return "CT_TVM_Model"

    def device(self):
        return self.model_device

    def box_proj(self, u):
        return torch.clamp(u, min=0.0, max=1.0)

    def D(self, u):
        batch_size = u.shape[-1]
        u_reshaped = u.view(128, 128, batch_size)
        Dux = torch.roll(u_reshaped, 1, 0) - u_reshaped
        Dux = Dux.view(128 ** 2, batch_size)
        Duy = torch.roll(u_reshaped, 1, 1) - u_reshaped
        Duy = Duy.view(128 ** 2, batch_size)
        Du = torch.cat((Dux, Duy), 0)
        return Du

    def Dt(self, p):
        batch_size = p.shape[-1]
        p_reshaped = p.view(2, 128, 128, batch_size)  # Split 2*128^2 into 2 x 128 x 128
        
        px = p_reshaped[0, :, :, :]  # Shape (128, 128, batch_size)
        Dtpx = torch.roll(px, -1, 0) - px
        Dtpx = Dtpx.view(128 ** 2, batch_size)

        py = p_reshaped[1, :, :, :]  # Shape (128, 128, batch_size)
        Dtpy = torch.roll(py, -1, 1) - py
        Dtpy = Dtpy.view(128 ** 2, batch_size)
        
        Dtp = Dtpx + Dtpy
        return Dtp

    def ball_proj(self, w, d, eps):
        dist = torch.norm(w - d, dim=0)
        dist[dist <= 1e-10] = 1e-10
        d_norm = torch.norm(d, dim=0)
        scale = torch.minimum(torch.ones(dist.shape, device=d.device),
                              self.eps * d_norm / dist)
        proj = d + scale * (w - d)
        return proj

    def forward(self, d, tol=1.0e-3, max_depth=500, depth_warning=False):
        self.depth = 0.0
        self.model_device = d.device
        
        # Ensure A and At are on the correct device
        A_dev = self.A.to(self.device())
        At_dev = self.At.to(self.device())

        d_vec = d.view(d.size()[0], -1).to(self.device())
        d_vec = d_vec.permute(1, 0)
        batch_size = d_vec.size()[1]
        
        uk = torch.zeros((128 ** 2, batch_size), device=self.device())
        pk = self.D(uk)
        wk = torch.matmul(A_dev, uk)
        nuk1 = torch.zeros(pk.size(), device=self.device())
        nuk2 = torch.zeros(d_vec.size(), device=self.device())

        for _ in range(max_depth):
            res1 = self.Dt(nuk1 + self.alpha * (self.D(uk) - pk))
            Auk = torch.matmul(A_dev, uk)
            res2 = torch.matmul(At_dev, nuk2 + self.alpha * (Auk - wk))
            rk = self.beta * (res1 + res2)
            uk = self.box_proj(uk - rk)

            res = self.lambd * (nuk1 + self.alpha * (self.D(uk) - pk))
            pk = self.shrink(pk + res)

            Auk = torch.matmul(A_dev, uk)
            res = self.lambd * (nuk2 + self.alpha * (Auk - wk))
            wk = self.ball_proj(wk + res, d_vec, self.eps)

            nuk1 = nuk1 + self.alpha * (self.D(uk) - pk)
            nuk2 = nuk2 + self.alpha * (torch.matmul(A_dev, uk) - wk)

        self.depth = max_depth
        if self.depth >= max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        uk = uk.permute(1, 0)
        return uk.view(uk.shape[0], 1, 128, 128)

# -------------------------------------------------
# 7. MODEL: Scale_CT_L2O_Model 
# -------------------------------------------------
class Scale_CT_L2O_Model(ImplicitL2OModel):
    def __init__(self,
                 A,
                 lambd=0.1,
                 alpha=0.1,
                 beta=0.1,
                 delta=-5.0,
                 K_out_channels=44,
                 max_depth=150): 
        super().__init__()
        self.A = A
        self.At = A.t()
        self.max_depth = max_depth
        self.K_out_channels = K_out_channels
        self.fixed_point_error = 0.0
        self.fidelity_rel_norm_error = 0.0

        # Non-trainable parameters
        self.delta_val = delta  # Renamed to avoid conflicts
        self.alpha_val = alpha
        self.lambd_val = lambd
        self.beta_val = beta
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Layers for R
        self.conv1 = nn.Conv2d(in_channels=K_out_channels,
                               out_channels=K_out_channels,
                               kernel_size=5, stride=1, padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=K_out_channels,
                               out_channels=K_out_channels,
                               kernel_size=5, stride=1, padding=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=K_out_channels,
                               out_channels=K_out_channels,
                               kernel_size=5, stride=1, padding=(2, 2))
        # Layers for K
        self.convK = nn.Conv2d(in_channels=1,
                               out_channels=K_out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.convK_T = nn.ConvTranspose2d(in_channels=K_out_channels,
                                          out_channels=1,
                                          kernel_size=3, padding=1, bias=False)

    def _get_conv_crit(self, x, x_prev, d, tol=1.0e-2):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        d = d.view(batch_size, -1)
        x_prev = x_prev.view(batch_size, -1)
        res_norm = torch.max(torch.norm(x - x_prev, dim=1))
        residual_conv = res_norm <= tol
        return residual_conv

    def name(self) -> str:
        return "Scale_CT_L2O_Model"

    def box_proj(self, x):
        return torch.clamp(x, min=0.0, max=1.0)

    def K(self, x):
        batch_size = x.shape[1]
        x = x.permute(1, 0).view(batch_size, 1, 128, 128)
        x = self.convK(x)
        x = x.view(batch_size, -1).permute(1, 0)
        return x

    def Kt(self, p):
        batch_size = p.shape[1]
        p = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)
        p = self.convK_T(p)
        p = p.view(batch_size, -1).permute(1, 0)
        return p

    def ball_proj(self, w, d, delta, proj_weight=0.99):
        # Since delta is not an nn.Parameter, use it directly
        delta_exp = torch.exp(torch.tensor(delta, device=self.device())) 
        dist = torch.norm(w - d, dim=0)
        d_norm = torch.norm(d, dim=0)
        dist[dist <= 1e-10] = 1e-10
        scale = torch.minimum(torch.ones(dist.shape, device=d.device),
                              delta_exp * d_norm / dist)
        proj = d + scale * (w - d)
        other = d + delta_exp * d_norm * (w - d) / dist

        if self.training:
            return proj_weight * proj + (1.0 - proj_weight) * other
        else:
            return proj

    def R(self, p):
        batch_size = p.shape[1]
        p_res = p.permute(1, 0).view(batch_size, self.K_out_channels, 128, 128)
        p_res = p_res + self.leaky_relu(self.conv1(p_res))
        p_res = p_res + self.leaky_relu(self.conv2(p_res))
        p_res = p_res + self.leaky_relu(self.conv3(p_res))
        p_res = p_res.view(batch_size, -1).permute(1, 0)
        p_res = p_res.view(self.K_out_channels * (128 ** 2), batch_size)
        return p_res
    
    def S(self, z):
        return torch.sign(z) * torch.maximum(torch.abs(z) - 0.1, torch.zeros_like(z))

    def _apply_T(self, x: inference, d: input_data, return_tuple=False):  # type: ignore
        batch_size = x.shape[0]

        d_vec = d.view(d.shape[0], -1).to(self.device())
        d_vec = d_vec.permute(1, 0)
        xk = x.view(x.shape[0], -1)
        xk = xk.permute(1, 0)
        pk = self.K(xk)
        wk = torch.matmul(self.A.to(self.device()), xk)
        nuk1 = torch.zeros(pk.size(), device=self.device())
        nuk2 = torch.zeros(d_vec.size(), device=self.device())

        # Use constant values
        alpha = torch.clamp(torch.tensor(self.alpha_val), min=0, max=2)
        beta = torch.clamp(torch.tensor(self.beta_val), min=0, max=2)
        lambd = torch.clamp(torch.tensor(self.lambd_val), min=0, max=2)
        delta = self.delta_val

        # pk step
        pk = pk + lambd * (nuk1 + alpha * (self.K(xk) - pk))
        pk = self.R(pk)

        # wk step
        Axk = torch.matmul(self.A.to(self.device()), xk)
        res_temp = nuk2 + alpha * (Axk - wk)
        temp_term = wk + lambd * res_temp
        wk = self.ball_proj(temp_term, d_vec, delta)

        # nuk1 step
        res_temp = self.K(xk) - pk
        nuk1_plus = nuk1 + alpha * res_temp

        # nuk2 step
        res_temp = Axk - wk
        nuk2_plus = nuk2 + alpha * res_temp

        # rk step
        self.convK_T.weight.data = self.convK.weight.data
        rk = self.Kt(2 * nuk1_plus - nuk1)
        rk = rk + torch.matmul(self.At.to(self.device()), 2 * nuk2_plus - nuk2)

        # xk step
        xk = torch.clamp(xk - beta * rk, min=0, max=1)
        
        xk_res = xk.permute(1, 0).view(batch_size, 1, 128, 128)

        if return_tuple:
            return xk_res, nuk1_plus, pk
        else:
            return xk_res

    def forward(self, d, depth_warning=False, return_depth=False, tol=1e-3, return_all_vars=False):
        with torch.no_grad():
            self.depth = 0.0
            x = torch.zeros((d.size()[0], 1, 128, 128),
                            device=self.device())
            x_prev = np.inf * torch.ones(x.shape, device=self.device())
            all_samp_conv = False

            while not all_samp_conv and self.depth < self.max_depth:
                x_prev = x.clone()
                x = self._apply_T(x, d)
                all_samp_conv = self._get_conv_crit(x,
                                                    x_prev,
                                                    d,
                                                    tol=tol)
                self.depth += 1

        if self.depth >= self.max_depth and depth_warning:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        self.fixed_point_error = torch.max(torch.norm(x - x_prev, dim=1))

        Tx, nuk1, pk = self._apply_T(x, d, return_tuple=True)

        if return_depth:
            return Tx, self.depth
        elif return_all_vars:
            return Tx, nuk1, pk
        else:
            return Tx
