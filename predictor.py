import torch
import utils

class Predictor:
    def __init__(self,
                 model,
                 data_loader,
                 device,
                 save_predictions_path):

        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.save_predictions_path = save_predictions_path

    def load_model_state(self, model_state_path):
        self.model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))

    def run(self, model_state_path):
        self.load_model_state(model_state_path)
        self.model.eval()
        predictions = []

        for image, label in self.data_loader:
            image = image.to(self.device)
            output = self.model(image)
            prediction = self.postprocess(output)
            predictions.append(prediction)
            print("label", label)
            print("predictions", prediction)
            print("eq", prediction.eq(label))
            print("sum", sum(prediction.eq(label)).numpy() / image.shape[0])

    def postprocess(self, output):
        label = torch.argmax(output, dim=1)
#        class_names = [utils.CLASS_NAMES[l] for l in label]
        return label

