import torch
import torch.nn as nn


class ViTTinyAutoencoder(nn.Module):
    def __init__(self, args):
        super(ViTTinyAutoencoder, self).__init__()

        # Calculate the number of patches
        self.num_patches = (args.image_size // args.patch_size) ** 2
        patch_dim = args.in_channel * args.patch_size * args.patch_size

        # Patch Embedding Layer (Encoder)
        self.encoder_patch_embedding = nn.Linear(patch_dim, args.hidden_dim)

        # Positional Encoding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, args.sequence_length * self.num_patches + 1, 
                        args.hidden_dim))

        # Transformer Encoder (Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_dim,
            nhead=args.n_heads,
            dim_feedforward=args.hidden_dim * 2
        )
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=args.transformer_layer)

        # Transformer Decoder (Decoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.hidden_dim,
            nhead=args.n_heads,
            dim_feedforward=args.hidden_dim * 2
        )
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer,
                                                         num_layers=args.transformer_layer)

        # Patch Embedding Layer (Decoder)
        self.decoder_patch_embedding = nn.Linear(args.hidden_dim, patch_dim)

    def forward(self, x):
        # Input size: (batch_size, channels, frames, height, width)
        batch_size, frames, channels, height, width = x.shape
        x = x.view(batch_size, frames * self.num_patches, -1)
        # Encoder
        x_encoder = self.encoder_patch_embedding(x)
        x_encoder += self.positional_embedding[:, : self.num_patches * frames]
        x_encoder = self.encoder_transformer(x_encoder)

        # Decoder
        x_decoder = self.decoder_transformer(x_encoder, x_encoder)
        x_decoder = self.decoder_patch_embedding(x_decoder)
        x_decoder = x_decoder.view(batch_size, frames, channels, height, width)

        x_encoder = x_encoder.view(batch_size, -1)

        return x_decoder, x_encoder


# model = ViTTinyAutoencoder(image_size=64, patch_size=16)
# input_data = torch.randn(16, 3, 4, 64, 64)
# output_decoder, output_encoder = model(input_data)
# print(output_decoder.shape)
# print(output_encoder.shape)
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# num_params = count_parameters(model)
# print(f"Number of trainable parameters: {num_params}")




