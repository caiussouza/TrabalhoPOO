from src.app.Breast_Cancer_APP import Breast_Cancer_APP
from src.models.GGRBF_BC_Classifier import GGRBF_BC_Classifier

model = GGRBF_BC_Classifier()
model.fit_model()
interface = Breast_Cancer_APP(model)
interface.start_app()
