from models import ModelInference

def main():
    print("Initializing model...")
    model = ModelInference()
    
    if model.initialized:
        print("Model initialized successfully!")
        model.test_model_connectivity()
    else:
        print("Model initialization failed")
        
if __name__ == "__main__":
    main()