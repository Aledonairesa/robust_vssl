from builtins import float
import argparse


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(
        'expected a boolean value: true/false, 1/0, or yes/no')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset_path', default='/', type=str, help='Root directory path of training data')
    parser.add_argument('--vggss_test_path', default='/',\
            type=str, help='Root directory path of data')
    parser.add_argument('--is3plus_test_path', default='data/IS3plus',
                        type=str, help='Root directory path of IS3plus data')
    parser.add_argument('--avsbench_test_path', default='data/AVSBench',
                        type=str, help='Root directory path of AVSBench data')
   
    parser.add_argument('--train_set_scale', default='fullset', type=str, help="fullset | subset_144k | subset_143k | subset_10k | subset_1k | subset_100")
    parser.add_argument('--test_set_scale', default='fullset', type=str, help="fullset | subset_250 | subset_50")
    parser.add_argument('--val_set_scale', default='subset_1k', type=str, help='validation set scale: subset_1k | subset_250')
    parser.add_argument('--test_set', default="VGGSS", type=str,
                        choices=['VGGSS', 'IS3plus', 'AVSBench'],
                        help='Testing set: VGGSS | IS3plus | AVSBench')

    parser.add_argument('--model_name', default='vgg', type=str, help='test files')
    
    parser.add_argument('--img_backbone_type', default='resnet18', type=str, help='resnet18 | dino_vit')
    parser.add_argument('--dino_model_name', default='facebook/dinov3-vitb16-pretrain-lvd1689m', type=str, help='DINO ViT model name from HuggingFace')
    parser.add_argument('--freeze_dino', type=parse_bool, default=True,
                        help='Whether to freeze the DINO backbone during training')
    parser.add_argument('--use_vision_blocks', action='store_true', help='Whether to use additional vision transformer blocks after DINO ViT backbone')
    parser.add_argument('--aud_backbone_type', default='resnet18', type=str,
                        choices=['resnet18', 'whisper', 'beats'], help='resnet18 | whisper | beats')
    parser.add_argument('--whisper_model_name', default='openai/whisper-base', type=str,
                        help='Whisper model name from HuggingFace')
    parser.add_argument('--freeze_whisper', type=parse_bool, default=True,
                        help='Whether to freeze the Whisper audio backbone during training')
    parser.add_argument('--beats_checkpoint', default='pretrained/beats/BEATs_iter3_plus_AS2M.pt', type=str,
                        help='Path to BEATs checkpoint')
    parser.add_argument('--freeze_beats', type=parse_bool, default=True,
                        help='Whether to freeze the BEATs audio backbone during training')
    parser.add_argument('--use_audio_blocks', action='store_true',
                        help='Whether to use additional audio transformer blocks after Whisper encoder')
    parser.add_argument('--tri_map',action='store_true')
    parser.set_defaults(tri_map=True)
    parser.add_argument('--Neg', action='store_true')
    parser.set_defaults(Neg=True)
    parser.add_argument('--cl_loss', default='ce', choices=['ce', 'ce_sym', 'sigmoid'],
                        help='Contrastive loss: original cross entropy, symmetric cross entropy, sigmoid (binary classification per logit)')
    parser.add_argument('--sigmoid_t_init', default=1.0, type=float,
                        help='Initial sigmoid-loss logit scale; optimized as exp(t)')
    parser.add_argument('--sigmoid_b_init', default=0.0, type=float,
                        help='Initial sigmoid-loss logit bias')
    parser.add_argument('--epsilon', default=0.65, type=float)
    parser.add_argument('--epsilon2', default=0.4, type=float)
    parser.add_argument('--batch_size', default=256, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=80, type=int, help='Number of total epochs to run')
    parser.add_argument('--image_size', default=224,type=int,help='Height and width of inputs')
    parser.add_argument('--learning_rate', default=1e-4,type=float,help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--output_dir', default='outputs', type=str,
                        help='Root directory for experiment outputs')
    parser.add_argument('--normalisation', default='all',type=str)
    parser.add_argument('--gpus', default="0", type=str, help='gpus')
    parser.add_argument('--pool', default="avgpool", type=str,help= 'pooling')
    parser.add_argument('--data_aug', action='store_true')
    parser.set_defaults(data_aug=True)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--n_threads', default=16, type=int,help='Number of threads for multi-thread loading')
    parser.add_argument('--epi_decay', action='store_true', help='two episons decay, no need for the experiment')
    parser.add_argument('--load_pretrain', action='store_true', help='Load pretrained model weights')
    parser.add_argument('--flow', action='store_true', help='  ' )
    parser.add_argument('--start_epoch', type=int, default=1, help="Start epoch for traing the model")

    parser.add_argument('--resume', type=str, default='', help='')
    parser.add_argument('--test', type=str, default='', help=' ')
    parser.add_argument('--eval_freq', type=int, default=1, help=' ')
    parser.add_argument('--eval_start', type=int, default=10, help='Epoch to start validation')
    parser.add_argument('--save_val_embeddings', action='store_true',
                        help='Save validation image/audio embeddings once per validation epoch')
    parser.add_argument('--save_test_embeddings', action='store_true',
                        help='Save test image/audio embeddings')
    parser.add_argument('--checkpoint_metric', default='auto', type=str,
                        choices=['auto', 'mean_ciou', 'loss', 'top1_i2a', 'top1_a2i'],
                        help='Metric for best checkpoint selection. auto uses mean_ciou when available, else loss.')
    parser.add_argument('--early_stop_patience', default=0, type=int,
                        help='Stop after this many validation checks without improvement. 0 disables early stopping.')
    parser.add_argument('--early_stop_min_delta', default=0.0, type=float,
                        help='Minimum checkpoint score improvement required to reset early stopping.')
    parser.add_argument('--print_freq', type=int, default=15, help=' ')
    parser.add_argument('--exp_name', type=str, default='experiment', help=' ')
    parser.add_argument('--hostname', type=str, default=None, help='show which machine the model is trained on ')
    parser.add_argument("--temperature", default=0.07, type=float, help='Temperature for logits, 0.02, 0.05, 0.07, 0.1')
    
    parser.add_argument("--seed", default=4, type=int, help='Seed for torch and numpy initlization: 0 1 2 3 4 ')
    
    parser.add_argument('--obs_start_epoch', type=int, default=5, help='Epoch to start oneline batch selection')
    parser.add_argument("--obs_drop_fraction", type=float, default=0.25, help='Drop fraction of barch sample when online batch selection')
    
    # Augmentations
    parser.add_argument("--img_aug", type=str, default=None, help='Image augmentations')
    parser.add_argument("--aud_aug", type=str, default=None, help='Audio augmentations')

    parser.add_argument('--heatmap_size', type=int, default=14, help='Heatmap size of the heatmap')
    
    parser.add_argument('--trans_equi_weight', type=float,  default=1.0, help='Weights')
    parser.add_argument('--lambda_atp', type=float, default=0.0,
                        help='Weight for the Align True Pairs modality-gap loss. 0 disables it.')
    parser.add_argument('--lambda_cu', type=float, default=0.0,
                        help='Weight for the Centroid Uniformity modality-gap loss. 0 disables it.')
    parser.add_argument('--atp_cu_image_embedding', default='positive_mask_mean',
                        choices=['positive_mask_mean', 'maxpool'],
                        help='Image embedding used by L_ATP and L_CU: localized positive-mask mean or global maxpool.')
    parser.add_argument('--atp_cu_start_epoch', type=int, default=1,
                        help='Epoch from which L_ATP/L_CU are enabled when their weights are > 0.')

    parser.add_argument('--lambda_trans_ts', type=float,  default=1.0, help='Weights for the transformation equivariance loss')
    parser.add_argument('--lambda_trans_cl', type=float,  default=1.0, help='Weights for the transformation CL loss')
    parser.add_argument('--batch_trans_ratio', type=float, default=0.3, help=' ')

    parser.add_argument('--lambda_rescale', type=float, default=0.2, help=' ')
    parser.add_argument('--rescale_start_epoch', type=int, default=5, help=' ')
    parser.add_argument('--rescale_factor', type=float, nargs='+', default= [0.85, 1.0], help='rescale factors')
    parser.add_argument('--rescale_prob', type=float, default= 0.8, help='rescale probility')

    parser.add_argument('--equi_loss_type', type=str, default='mse', help='Loss type: l1loss: "l1loss"| mae | l2 loss: "mse" ')
    parser.add_argument('--max_rotation_angle', type=float, default=45)
    parser.add_argument('--biCLLoss', action='store_true', default=False)
    parser.add_argument('--heatmap_no_grad', action='store_true', default=False)
    parser.add_argument('--audio_extract_batch_size', type=int, default=256, help='batch size for extract audio embeddings ')
    parser.add_argument('--audio_queue_size', type=int, default=4000, help='The size of the queue for audio retrieval')
    parser.add_argument('--retri_save_dir', type=str, default='assets/retri/', help='save the closed audio for each audio, in a dict form')
    parser.add_argument('--ret_start_epoch', type=int, default=1, help='epcoh to start audio retrieval and replacment')
    parser.add_argument('--audio_replace_prob', type=float, default=-1.0, help='probability to replace the audio')
    parser.add_argument('--epoch', type=int, default=1, help='record current epoch')

    parser.add_argument('--audio_mix_alpha', type=float, default=-1, help='alpha for audio mix, ')
    parser.add_argument('--mix_start_epoch', type=int, default=1, help='epcoh to start audio retrieval and mix')
    parser.add_argument('--audio_mix_prob', type=float, default=-1, help='probability to mix the audio')
    parser.add_argument('--mix_curri_end_epoch', type=int, default=40, help='epcoh to start audio retrieval and mix')

    parser.add_argument('--ret_seen_144k', action='store_true', default=False)

    return parser.parse_args()
