import os
import torch

def get_available_vram():
    """Get available GPU memory in GB"""
    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
        return free_memory
    return 0

def select_model():
    """Allow user to select from available models based on their hardware"""
    
    available_vram = get_available_vram()
    print(f"\n=== Model Selection ===")
    print(f"Detected GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Available VRAM: {available_vram:.2f} GB\n")
    
    # Define available models with their requirements and specialties
    models = [
        {
            "name": "microsoft/phi-2", 
            "display_name": "Microsoft Phi-2",
            "vram_required": 4.0, 
            "specialty": "Lightweight, good balance of performance and speed",
            "type": "general"
        },
        {
            "name": "google/gemma-2b", 
            "display_name": "Google Gemma 2B",
            "vram_required": 3.0, 
            "specialty": "Very efficient, good for basic tasks",
            "type": "general"
        },
        {
            "name": "mistralai/Mistral-7B-Instruct-v0.2", 
            "display_name": "Mistral 7B",
            "vram_required": 8.0, 
            "specialty": "High accuracy for instruction following",
            "type": "general"
        },
        {
            "name": "meta-llama/Llama-3-8B-Instruct", 
            "display_name": "Meta Llama 3 (8B)",
            "vram_required": 12.0, 
            "specialty": "Advanced reasoning, best quality results",
            "type": "general"
        },
        {
            "name": "microsoft/deberta-v3-base", 
            "display_name": "DeBERTa-v3 (classification specialist)",
            "vram_required": 2.0, 
            "specialty": "Specialized for column classification only",
            "type": "classification"
        }
    ]
    
    # Filter models based on available VRAM
    suitable_models = []
    for i, model in enumerate(models):
        is_suitable = available_vram >= model["vram_required"] or not torch.cuda.is_available()
        suitable_models.append((i, model, is_suitable))
    
    # Display options
    print("Available models:")
    for i, model, is_suitable in suitable_models:
        status = "✓ " if is_suitable else "⚠ "
        vram_note = f" (Requires {model['vram_required']:.1f}+ GB VRAM)" if not is_suitable else ""
        print(f"{i+1}. {status}{model['display_name']}{vram_note}")
        print(f"   Specialty: {model['specialty']}")
    
    # Get user selection
    while True:
        try:
            selection = int(input("\nSelect model number (or 0 for default): ")) - 1
            if selection == -1:  # Default option (0)
                print("Using default model: Microsoft Phi-2")
                return "microsoft/phi-2"
            elif 0 <= selection < len(models):
                model = models[selection]
                print(f"Selected: {model['display_name']}")
                if not suitable_models[selection][2]:
                    print("\n⚠️ Warning: This model may not run well on your hardware!")
                    confirm = input("Continue anyway? (y/n): ").lower()
                    if confirm != 'y':
                        continue
                return model["name"]
            else:
                print("Invalid selection, please try again.")
        except ValueError:
            print("Please enter a number.")

if __name__ == "__main__":
    # Test the selection function
    selected_model = select_model()
    print(f"Selected model: {selected_model}")