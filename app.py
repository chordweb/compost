from flask import Flask, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

app = Flask(__name__)
CORS(app)

#default setup needed when this file is reloaded
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def do_ml(bio, groups):
    # groups = [{description: "", _id: ""}]
    # bio = ""
    # return an array of ids
    my_embedding = model.encode(bio, convert_to_tensor=True)
    similarities = []
    for group in groups:
        group_embedding = model.encode(group["description"], convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(bio_embedding, my_embedding)
        similarities.append([similarity_score, group["_id"]])
    sorted_groups = sorted(similarities)
    sorted_group_ids = [id for similarity, id in sorted_groups]
    sorted_group_ids.reverse()
    return sorted_group_ids

def differentially_private_training(pairs, scores):
    #adapts model in differentially private way to better match people going forward
    #pairs = [[str, str]]
    #scores = [double] (between a 0 and a 1 with a 0 being a bad match and a 1 being a good match)
    #pairs and scores must have same length

    #construct DataLoader
    train_examples = [InputExample(texts=[pair[0], pair[1]], label=score) for pair, score in zip(pairs, scores)]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

    #add in loss function
    train_loss = losses.CosineSimilarityLoss(model)

    #construct optimizers on our end
    optimizers = model.construct_optimizers(train_objectives=[(train_dataloader, train_loss)])

    #Make everything differentially private using the privacy engine
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
    )

    #train!
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, specified_optimizers=optimizers)




@app.route("/", methods=["POST"])
def hello_world():
    bio = request.json["bio"]
    groups = request.json["groups"]
    res = do_ml(bio, groups)
    return {"matches": res}

app.run(port=7070)
