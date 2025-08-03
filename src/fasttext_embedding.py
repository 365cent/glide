    parser = argparse.ArgumentParser(description="Generate FastText embeddings for log data.")
    parser.add_argument("--log_type", type=str, required=True,
                        help="Specific log type to process (e.g., 'vpn', 'wp-error').")
    parser.add_argument("--use_global_attack_list", action='store_true',
                        help="Use a global list of attack types across all log types.")
    
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting FastText embedding process for log type: {args.log_type}")

    # 1. Load data
    try:
        df = load_tfrecord_files(log_type_filter=args.log_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Display initial data distribution
    display_data_distribution(df, args.log_type)

    # 2. Load pre-trained FastText model
    fasttext_model = load_pretrained_fasttext()
    if fasttext_model is None:
        print("Failed to load FastText model. Exiting.")
        return

    # 3. Process embeddings and binary labels
    df = process_embeddings(df, fasttext_model, use_global_attack_list=args.use_global_attack_list)

    # 4. Save embeddings and labels
    log_embeddings_path = OUTPUT_DIR / args.log_type / f"log_{args.log_type}.pkl"
    label_data_path = OUTPUT_DIR / args.log_type / f"label_{args.log_type}.pkl"
    attack_types_path = OUTPUT_DIR / args.log_type / f"attack_types_{args.log_type}.txt"

    (OUTPUT_DIR / args.log_type).mkdir(parents=True, exist_ok=True)

    with open(log_embeddings_path, 'wb') as f:
        pickle.dump(np.array(df['log_embedding'].tolist()), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Log embeddings saved to {log_embeddings_path}")

    # Prepare label data for saving
    if args.use_global_attack_list:
        all_attack_types = df.attrs['attack_types']
        label_vectors = np.array(df['binary_labels'].tolist())
    else:
        # For per-log-type processing, we need to handle the varying dimensions of binary_labels
        # This means we need to save attack_types per log_type, which is already handled by df.attrs
        # We need to ensure that all binary_labels in the DataFrame for this log_type have the same dimension
        # This implies that `process_embeddings` should have created consistent binary_labels for this log_type
        # which it does by using `log_type_to_attacks[log_type]`
        
        # Get the attack types for the current log_type
        current_log_type_attacks = df.attrs['log_type_to_attacks'][args.log_type]
        label_vectors = np.array(df['binary_labels'].tolist())
        all_attack_types = current_log_type_attacks

    label_data = {
        'vectors': label_vectors,
        'classes': all_attack_types,
        'description': 'Binary multi-label vectors where [0 1 0] means only the second class is present'
    }
    with open(label_data_path, 'wb') as f:
        pickle.dump(label_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Label data saved to {label_data_path}")

    with open(attack_types_path, 'w') as f:
        f.write("Attack Types:\n")
        for i, attack_type in enumerate(all_attack_types):
            f.write(f"  {i}: {attack_type}\n")
    print(f"Attack types mapping saved to {attack_types_path}")

    # 5. Visualize embeddings
    visualization_output_path = OUTPUT_DIR / args.log_type / f"tsne_visualization_{args.log_type}.png"
    visualize_embeddings(df, output_file=visualization_output_path)

    print(f"FastText embedding process for {args.log_type} completed.")

if __name__ == '__main__':