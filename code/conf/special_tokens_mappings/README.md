## Special Tokens

Since the special tokens differ from task to task, they are stored in a JSON file in the `data/special_tokens_mappings` directory.

To use the correct special tokens mapping, make sure that `directories.special_tokens_mappings` in the config points to the correct directory and `dataset.special_tokens_file` points to the correct file.

> [!TIP]
> To simplify the process, by default the code assumes that the special tokens file is named `special_tokens_<task_name>.json` and is located in the `data/special_tokens_mappings` directory.