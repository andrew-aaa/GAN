from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
LOG_DIR = BASE_DIR / 'logs'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ===== sequence =====
MAX_AA_LEN = 256
MAX_LEN = MAX_AA_LEN + 2  # BOS + up to MAX_AA_LEN tokens/EOS target length convention
PAD_TOKEN = '_'
BOS_TOKEN = '^'
EOS_TOKEN = '*'
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
VOCAB = PAD_TOKEN + BOS_TOKEN + EOS_TOKEN + AMINO_ACIDS
VOCAB_SIZE = len(VOCAB)
PAD_IDX = VOCAB.index(PAD_TOKEN)
BOS_IDX = VOCAB.index(BOS_TOKEN)
EOS_IDX = VOCAB.index(EOS_TOKEN)
AA_START_IDX = VOCAB.index('A')

# ===== model =====
ESM_DIM = 320
LATENT_DIM = 64
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
FF_MULT = 4
DROPOUT = 0.1
DISC_EMBED_DIM = 128
DISC_NUM_HEADS = 4
DISC_NUM_LAYERS = 2
DISC_DROPOUT = 0.0
PROJECTION_DIM = 128

# ===== training =====
SEED = 42
# Для Colab T4 лучше 2-4. Если памяти хватает, можно поднять до 8/16.
BATCH_SIZE = 4
EPOCHS = 40
GENERATOR_PRETRAIN_EPOCHS = 8
N_CRITIC = 3
VAL_SPLIT = 0.15
LR_G = 1e-4
LR_D = 3e-4
BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
LAMBDA_GP = 10.0
ADV_WEIGHT_MAX = 0.10
MISMATCH_WEIGHT = 0.5

# Основные supervised-компоненты
TOKEN_CE_WEIGHT = 1.0
LENGTH_LOSS_WEIGHT = 0.75
LABEL_SMOOTHING = 0.0

# Явный контроль длины/EOS/PAD. Эти веса добавлены после анализа графиков:
# nonempty высокий, но eos_exact был около нуля, а len_mae был большим.
EOS_LOSS_WEIGHT = 0.75
PAD_AFTER_EOS_WEIGHT = 0.35
AA_BEFORE_EOS_WEIGHT = 0.25
BOS_FORBIDDEN_WEIGHT = 0.05

# Весовые коэффициенты только для выбора best-checkpoint по val_proxy.
PROXY_LENGTH_WEIGHT = 0.50
PROXY_EOS_WEIGHT = 0.50
PROXY_PAD_WEIGHT = 0.20
PROXY_AA_KL_WEIGHT = 0.20

EMA_DECAY = 0.999
TAU_START = 1.0
TAU_END = 0.5
NUM_WORKERS = 0
PIN_MEMORY = False
RESET_METRICS_CSV = True
EMPTY_CACHE_EACH_EPOCH = False

# ===== paths =====
TOXIN_FASTA_PATH = str(DATA_DIR / 'toxins_paired.fasta')
ANTITOXIN_FASTA_PATH = str(DATA_DIR / 'antitoxins_paired.fasta')
RAW_TOXIN_FASTA_PATH = str(DATA_DIR / 'type_II_T_exp.fas')
RAW_ANTITOXIN_FASTA_PATH = str(DATA_DIR / 'type_II_AT_exp.fas')
TOXIN_EMBEDDINGS_PATH = str(DATA_DIR / 'toxin_embeddings.pt')

GENERATOR_LAST_PATH = str(CHECKPOINT_DIR / 'generator_last.pt')
GENERATOR_BEST_PATH = str(CHECKPOINT_DIR / 'generator_best_val.pt')
EMA_LAST_PATH = str(CHECKPOINT_DIR / 'generator_ema_last.pt')
EMA_BEST_PATH = str(CHECKPOINT_DIR / 'generator_ema_best_val.pt')
METRICS_CSV_PATH = str(LOG_DIR / 'metrics.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== validation / inference =====
# Валидация теперь дополнительно оценивает реальный режим генерации:
# длина берётся из length-head, а EOS/PAD жёстко маскируются по этой длине.
CONSTRAINED_EVAL = True
TOP_K = 8
NUM_GENERATION_ATTEMPTS = 16
GENERATION_TEMPERATURES = (1.2, 1.0, 0.9, 0.8)
