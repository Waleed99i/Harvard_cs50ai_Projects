import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL, from_pt=True)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)



def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """

    input_ids = list(inputs['input_ids'].numpy()[0])
    try:
        index = input_ids.index(mask_token_id)
    except ValueError:
        return None
    
    return index


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    rgb = (255 * attention_score.numpy())//1
    return (int(rgb), int(rgb), int(rgb))

'''
my above code converts an attention score between 0 and 1 to an RGB color.

    0 --> (0, 0, 0) (black)

    1 --> (255, 255, 255) (white)

  0.5 --> (127, 127, 127) (gray)


| Attention Score | Meaning          | Color |
| --------------- | ---------------- | ----- |
| 0.0             | No attention     | Black |
| 1.0             | Full attention   | White |
| 0.5             | Medium attention | Gray  |

letme explain u with an example 

If attention score = x (between 0 and 1):

gray_value = int(x * 255)
rgb = (gray_value, gray_value, gray_value)

Examples:

    x = 0 → (0, 0, 0) → black

    x = 1 → (255, 255, 255) → white

    x = 0.25 → (64, 64, 64) → dark gray

    x = 0.75 → (191, 191, 191) → light gray

here basically x is the attention score, which is a float value between 0 and 1. The function converts this score into a shade of gray by multiplying it by 255, which is the maximum value for an RGB color channel. The resulting value is then used to create an RGB tuple where all three channels (red, green, blue) have the same value, resulting in a shade of gray.
'''



def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """

    for i in range(len(attentions)):       # For each layer (12 total)
        for k in range(len(attentions[i][0])):  # For each head (12 per layer)
            generate_diagram(
                i+1,                        # layer_number (start from 1)
                k+1,                        # head_number (start from 1)
                tokens,                    # word tokens like ["the", "cat", ...]
                attentions[i][0][k]        # the actual attention matrix
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"attentions/Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()






