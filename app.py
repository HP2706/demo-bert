from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from potassium import Potassium, Request, Response

app = Potassium("my_app")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

@app.init
def init():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device).eval()

    context = {
        "tokenizer": tokenizer,
        "model": model
    }

    return context

@app.handler()
def handler(context: dict, request: Request) -> Response:
    sentences = request.json.get("sentences")
    tokenizer = context.get("tokenizer")
    model = context.get("model")

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(model.device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return Response(
        json={"embeddings": sentence_embeddings.tolist()},
        status=200
    )

if __name__ == "__main__":
    app.serve()