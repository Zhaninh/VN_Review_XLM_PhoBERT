from fastapi import FastAPI, Request
from pydantic import BaseModel
from datasets import DatasetDict, Dataset
from fastapi.templating import Jinja2Templates
import torch
from models import CustomXLMModel
from helpers import pred_to_label
from preprocessing import preprocess

app = FastAPI()
templates = Jinja2Templates(directory='templates')

# Khai bao
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = r"D:\FSoft\Review_Ana\Dream_Tim\A\weights\XLM\model.pt"


model = CustomXLMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))
model.to(device)
model.eval()


class Review(BaseModel):
    text: str


@app.post("/analyse")
async def analyse_review(review_data: Review):
    review_sentence = (review_data.text,)

    review_dict = {"Review": review_sentence}
    input_data = Dataset.from_dict(review_dict)
    prep = preprocess('xlm-roberta-base')
    preprocessed_data = prep.run_test(DatasetDict({"test": input_data}))

    test_df = preprocessed_data['test']
    inputs = {'input_ids': test_df['input_ids'].to(device),
              'attention_mask': test_df['attention_mask'].to(device)}

    with torch.no_grad():
        outputs_classifier, outputs_regressor = model(**inputs)
        outputs_classifier = outputs_classifier.cpu().numpy()
        outputs_regressor = outputs_regressor.cpu().numpy()
        outputs_regressor = outputs_regressor.argmax(axis=-1) + 1

        outputs = pred_to_label(outputs_classifier, outputs_regressor)

        return {
            "Giai_tri": int(outputs[0][0])*"⭐ ",
            "Luu_tru": int(outputs[0][1])*"⭐ ",
            "Nha_hang": int(outputs[0][2])*"⭐ ",
            "An_uong": int(outputs[0][3])*"⭐ ",
            "Di_chuyen": int(outputs[0][4])*"⭐ ",
            "Mua_sam": int(outputs[0][5])*"⭐ "
        }

@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
