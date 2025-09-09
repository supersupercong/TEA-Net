from .loss_vif import fusion_loss_vif
from .loss_vif1 import fusion_loss_vif1
from .loss_med import fusion_loss_med
from .loss_mff import fusion_loss_mff
from .loss_nir import fusion_loss_nir
from .highorder_nips import Net

MODELS = {
          "highorder_nips":Net,
            }

LOSSES = {
          "Loss_Vif":fusion_loss_vif,
          "Loss_Vif1":fusion_loss_vif1,
          "Loss_Med":fusion_loss_med,
          "Loss_Mff":fusion_loss_mff,
          "Loss_Nir":fusion_loss_nir,
}