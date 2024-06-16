from src.app.Breast_Cancer_APP import Breast_Cancer_APP
from src.models.GGRBF_BC_Classifier import GGRBF_BC_Classifier

# Instantiate the model
model = GGRBF_BC_Classifier()
# Fit the model
model.fit_model()

# Instantiate the interface based on the model
interface = Breast_Cancer_APP(model)
# Start the interface
interface.start_app()
