import argparse
import os
import pickle
from glob import glob

import pandas as pd
from dtds.data.load import load_dataset
from dtds.eval.distribution_analysis import distribution_analysis
from dtds.features.transformers import decode_train_data
from dtds.synthesizers.ctgan import CTGANSynthesizer

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", default="loan", choices=["loan", "berka"])
parser.add_argument("-epochs", default=3, type=int)
parser.add_argument("-report", action="store_true")
args = parser.parse_args()


meta_path = glob(f"data/processed/{args.dataset}*/*.json")[0]
model_name = f"{args.dataset}_epoch{args.epochs}"
model_path = f"models/synthesizer_{model_name}.pkl"


train, test, meta, categoricals, ordinals, bimodals, label_encoders = load_dataset(args.dataset, benchmark=True)


if not os.path.exists(model_path) or not args.report:
    synthesizer = CTGANSynthesizer(epochs=args.epochs)
    synthesizer.fit(train, categoricals, ordinals, bimodals)
    with open(model_path, "wb") as f:
        pickle.dump(synthesizer, f, -1)
else:
    synthesizer = pickle.load(open(model_path, "rb", -1))


if args.report:
    synthesized = synthesizer.sample(1000)

    synthesized = decode_train_data(synthesized, meta_path, label_encoders, model_name=model_name, save=False)
    real = decode_train_data(test, meta_path, label_encoders, model_name=model_name, save=False)
    print(synthesized)
    print(real)
    distribution_analysis(
        real,
        synthesized,
        meta_path=meta_path,
        saving_path=f"reports/proof_of_concept/{model_name}/",
    )
