from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ===== Sequence setup =====
MAX_AA_LEN = 256                     # максимум аминокислот без учёта EOS
MAX_LEN = MAX_AA_LEN + 1             # целевая длина с EOS

PAD_TOKEN = "_"
BOS_TOKEN = "^"
EOS_TOKEN = "*"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = PAD_TOKEN + BOS_TOKEN + EOS_TOKEN + AMINO_ACIDS
VOCAB_SIZE = len(VOCAB)

PAD_IDX = VOCAB.index(PAD_TOKEN)
BOS_IDX = VOCAB.index(BOS_TOKEN)
EOS_IDX = VOCAB.index(EOS_TOKEN)
AA_START_IDX = VOCAB.index("A")

# ===== Model params =====
LATENT_DIM = 64
EMBED_DIM = 192
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

DISC_EMBED_DIM = 128
DISC_NUM_HEADS = 4
DISC_NUM_LAYERS = 2
DISC_DROPOUT = 0.0                  # по аудиту: убрать dropout из D для стабильности WGAN-GP
PROJECTION_DIM = 192                # размерность projection-discriminator

ESM_DIM = 320  # esm2_t6_8M_UR50D

# ===== Training =====
BATCH_SIZE = 8
EPOCHS = 80
GENERATOR_PRETRAIN_EPOCHS = 12
N_CRITIC = 4
VAL_SPLIT = 0.15
SEED = 42

LR_G = 1e-4
LR_D = 3e-4                         # TTUR: critic обучается быстрее
BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-4
LAMBDA_GP = 10.0
MISMATCH_WEIGHT = 0.5
EMA_DECAY = 0.999
GRAD_CLIP_NORM = 1.0

ADV_WEIGHT_MAX = 0.20
TOKEN_CE_WEIGHT = 1.0
LENGTH_LOSS_WEIGHT = 0.25
PAD_WEIGHT = 0.15
LABEL_SMOOTHING = 0.0

TAU_START = 1.0                     # annealing для Gumbel-Softmax в обучении
TAU_END = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Paths =====
MODEL_SAVE_PATH = str(CHECKPOINT_DIR / "generator_last.pt")
BEST_PROXY_MODEL_SAVE_PATH = str(CHECKPOINT_DIR / "generator_best_proxy.pt")
EMA_MODEL_SAVE_PATH = str(CHECKPOINT_DIR / "generator_ema_last.pt")
BEST_EMA_PROXY_MODEL_SAVE_PATH = str(CHECKPOINT_DIR / "generator_ema_best_proxy.pt")
METRICS_CSV_PATH = str(LOG_DIR / "metrics.csv")

TOXIN_FASTA_PATH = str(DATA_DIR / "toxins_paired.fasta")
ANTITOXIN_FASTA_PATH = str(DATA_DIR / "antitoxins_paired.fasta")
RAW_TOXIN_FASTA_PATH = str(DATA_DIR / "type_II_T_exp.fas")
RAW_ANTITOXIN_FASTA_PATH = str(DATA_DIR / "type_II_AT_exp.fas")
TOXIN_EMBEDDINGS_PATH = str(DATA_DIR / "toxin_embeddings.pt")
