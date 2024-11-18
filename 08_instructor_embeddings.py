from InstructorEmbedding import INSTRUCTOR
from decouple import config
from huggingface_hub import login

api_token = config("HUGGING_FACE_TOKEN")
login(token=api_token)


# sentence-transformers==2.2.2 InstructorEmbedding==1.0.1

if __name__ == "__main__":
    model = INSTRUCTOR("hkunlp/instructor-base")

    sentence = "RAG Fundamentals First"

    instruction = "Represent the title of an article about AI:"

    embeddings = model.encode([[instruction, sentence]])
    print(embeddings.shape)  # noqa
    # Output: (1, 768)