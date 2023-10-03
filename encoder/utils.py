import json



def get_model_files(args):
    checkpoint_dir = args.checkpoint_file.format("last", args.image_size, 
                                                 args.patch_size, args.sequence_length, 
                                                 args.in_channel, args.hidden_dim, 
                                                 args.n_heads, args.transformer_layer)

    best_model_dir = args.checkpoint_file.format("best", args.image_size,
                                                 args.patch_size, args.sequence_length,
                                                 args.in_channel, args.hidden_dim, 
                                                 args.n_heads, args.transformer_layer)

    return checkpoint_dir, best_model_dir


def read_url_files(key):
    with open('encoder/config/data.json', 'r') as json_file:
        data = json.load(json_file)

    return data["DataUrl"][key]