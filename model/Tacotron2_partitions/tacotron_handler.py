import torch

class MyCustomHandler(BaseHandler):
    def __init__(self):
        super(MyCustomHandler, self).__init__()
        
        
    
    def preprocess(self, data):
        # Preprocess the input data
        inputs = []
        for row in data:
            text = row.get("body")
            inputs.append(text)
            print(f"the input text: {text}")
        return torch.stack(inputs)

    def postprocess(self, data):
        # Postprocess the output data
        return data

