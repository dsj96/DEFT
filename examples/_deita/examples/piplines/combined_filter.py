import logging
import argparse
from deita.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--other_data_path", type=str, default=None)
parser.add_argument("--threshold", type=float, default=0.9)
parser.add_argument("--data_size", type=int, default=10)
parser.add_argument("--chunk_size", type=int, default=100000)
parser.add_argument("--sort_key", type=str, default="complexity_scores,quality_scores,knowledge20_scores,loss,knowledge5_scores,knowledge40_scores,knowledge10_scores,knowledge30_scores,knowledge60_scores,ppl,knowledge50_scores")
parser.add_argument("--output_path", type=str, default=None) # df["complexity_scores"] * df["quality_scores"] + df["quality_scores"]*df[all_sort_keys[2]]
parser.add_argument("--distance_metric", type=str, default="cosine")
parser.add_argument("--embedding_field", type=str, default="embedding")
parser.add_argument("--is_compression", type=bool, default=False)
parser.add_argument("--device", type=int, default="0")
parser.add_argument("--decay_rate", type=float, default=0.5)

args = parser.parse_args()

filter_pipeline = Pipeline("filter_pipeline", 
                          data_path = args.data_path,  # json file with sharegpt format
                          other_data_path = args.other_data_path,  # embedding file path (pickle format)
                          threshold = args.threshold,  # filter threshold default: 0.9 
                          data_size = args.data_size,  # size of selected data
                          chunk_size = args.chunk_size,  # used for more efficient GPU computing  default: 100000
                          sort_key = args.sort_key,  # default: "complexity_scores,quality_scores"
                          output_path = args.output_path,  # json format output path
                          distance_metric = args.distance_metric,  # default: cosine
                          embedding_field = args.embedding_field,  # default: embedding
                          is_compression = args.is_compression,  # default: False
                          device = args.device,  # GPU IDX, default: 0
                          decay_rate = args.decay_rate
                          )

filter_pipeline.run()