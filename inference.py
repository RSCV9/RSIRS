import torch
import torch.utils.data
import utils
import numpy as np
import transforms as T
import random
from bert.modeling_bert import BertModel
from lib import segmentation
from bert.tokenization_bert import BertTokenizer
from PIL import Image, ImageDraw, ImageFont



def overlay_mask_on_image(original_image_path, mask_array, output_image_path, final_size=(800, 450), text=''):

    original_image = Image.open(original_image_path).convert("RGBA")
    original_image_resized = original_image.resize((480, 480))

    red_overlay = np.zeros_like(np.array(original_image_resized))
    red_overlay[:, :, 0] = 255
    red_overlay[:, :, 3] = (mask_array * 255).astype(np.uint8)

    red_overlay_image = Image.fromarray(red_overlay, 'RGBA')
    combined_image = Image.alpha_composite(original_image_resized, red_overlay_image)

    width, height = combined_image.size
    border_height = 30
    new_image = Image.new("RGBA", (width, height + border_height), (0, 0, 0, 255))
    new_image.paste(combined_image, (0, border_height))

    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_position = ((width - text_width) // 2, (border_height - text_height) // 2)

    draw.text(text_position, text, font=font, fill=(255, 255, 255, 255))
    final_image = new_image.resize(final_size)
    final_image.save(output_image_path)


def evaluate(model, data, vis_cfg, bert_model, device):
    model.eval()

    with torch.no_grad():
        image, sentences, attentions = data[0], data[1], data[2]
        image, sentences, attentions = image.to(device), sentences.to(device), attentions.to(device)


        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)


        for j in range(sentences.size(-1)):
            if bert_model is not None:
                last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
            else:
                output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

            output = output.cpu()


            output_mask = output.argmax(1).data.numpy()[0]


        overlay_mask_on_image(vis_cfg[0], output_mask, vis_cfg[2], final_size=(800, 450), text=vis_cfg[1])

    print('已完成推理')
    print('图像已保存至：{}'.format(vis_cfg[2]))


def main(args):

    args.window12 = True

    args.swin_type = 'base'
    # ==========================================

    image_path = args.infer_img_path
    language = 'the ship on the left'  
    image_save_path = args.infer_img_savepath
    vis_cfg = [image_path, language, image_save_path]

    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)


    raw_img = Image.open(image_path).convert('RGB')


    target_size = (args.img_size, args.img_size)
    raw_img = raw_img.resize(target_size, Image.BILINEAR)


    img_np = np.array(raw_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_tensor = (img_tensor - mean) / std


    img = img_tensor.unsqueeze(0)


    max_tokens = 20

    sentences_for_ref = []
    attentions_for_ref = []

    sentence_raw = language
    attention_mask = [0] * max_tokens
    padded_input_ids = [0] * max_tokens

    input_ids = tokenizer.encode(text=sentence_raw, add_special_tokens=True)
    input_ids = input_ids[:max_tokens]

    padded_input_ids[:len(input_ids)] = input_ids
    attention_mask[:len(input_ids)] = [1] * len(input_ids)

    sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
    attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

    embedding = []
    att = []
    for s in range(len(sentences_for_ref)):
        e = sentences_for_ref[s]
        a = attentions_for_ref[s]
        embedding.append(e.unsqueeze(-1))
        att.append(a.unsqueeze(-1))

    tensor_embeddings = torch.cat(embedding, dim=-1)
    attention_mask = torch.cat(att, dim=-1)

    data = [img, tensor_embeddings, attention_mask]

    device = torch.device(args.device)

    single_model = segmentation.__dict__[args.model](pretrained='', args=args)

    checkpoint = torch.load(args.resume, map_location='cpu')

    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(model, data, vis_cfg, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser

    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
