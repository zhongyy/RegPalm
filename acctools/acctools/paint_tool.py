import numpy as np
from PIL import Image, ImageDraw, ImageFont
#from IPython import embed
from tqdm import tqdm
import os

def auto_pad(img, target_width, target_height, fill=0):
    raw_width, raw_height = img.size
    # black padding
    padding = Image.new('RGB', (target_width, target_height), (fill, fill, fill))
    if (float(raw_width) / raw_height) > ((float(target_width) / target_height)):
        caculated_height = round(float(target_width) / raw_width * raw_height)
        img = img.resize((target_width, caculated_height), Image.BILINEAR)
        pad_size = int((target_height - caculated_height) / 2)
        padding.paste(img, (0, pad_size))
    else:
        caculated_width = round(float(target_height) / raw_height * raw_width)
        img = img.resize((caculated_width, target_height), Image.BILINEAR)
        pad_size = int((target_width - caculated_width) / 2)
        padding.paste(img, (pad_size, 0))
    
    return padding

def label_paint(src, save_path, font_size=11, col_interval=10, pic_size=100,
        fixed_rows=None, fixed_cols=None, paint_label=True, fill=0):
    r"""
        src should be a list of tupele.
        each tuple has (
            img -- PIL.Image
            label -- string
            color -- string, of label
            row_idx -- int
            col_idx -- int
        )
        each image must be of the same size.
    """
    if paint_label:
        assert(font_size >= 5)
    #font_ttf_file = '/usr/share/fonts/dejavu/DejaVuSans.ttf'
    #font_ttf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DejaVuSans.ttf")
    #print("found font ttf file at {}".format(font_ttf_file))
    font_ttf_file = './DejaVuSans.ttf'
    font = ImageFont.truetype(font_ttf_file, font_size)
    rows = max([i[3] for i in src]) + 1
    cols = max([i[4] for i in src]) + 1
    if fixed_rows:
        assert(fixed_rows >= rows)
        rows = fixed_rows
    
    if fixed_cols:
        assert(fixed_cols >= cols)
        cols = fixed_cols
    

    #img_size = src[0][0].size
    img_size = (pic_size, pic_size)
    label_size = (img_size[0], font_size*2)
    canvas = np.ones((
        rows * (img_size[1] + label_size[1]),
        cols * (img_size[0] + col_interval) - col_interval,
        3), dtype=np.uint8) * 255
    print(canvas.shape)
    print(cols)
    print(rows)

    for item in tqdm(src):
        label_row = item[3] * (img_size[1] + label_size[1])
        label_col = item[4] * (img_size[0] + col_interval)
        image_row = label_row + label_size[1]
        image_col = label_col
        canvas[image_row : image_row + img_size[1],
            image_col : image_col + img_size[0]
            ] = np.array(
                auto_pad(item[0].convert('RGB'), pic_size, pic_size, fill=fill), 
                dtype=np.uint8)
        
        if paint_label:
            label_canvas = Image.fromarray(canvas[label_row : label_row + label_size[1],
                label_col : label_col + label_size[0]])
            drawobj=ImageDraw.Draw(label_canvas)
            drawobj.text([4, label_size[1]/2-2], item[1], item[2], font=font)
            canvas[label_row : label_row + label_size[1],
                label_col : label_col + label_size[0]] = np.array(label_canvas, dtype=np.uint8)
        
    canvas = Image.fromarray(canvas).convert('RGB')
    canvas.save(save_path)