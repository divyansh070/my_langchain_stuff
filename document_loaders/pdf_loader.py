from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('/Users/divyanshverma/Desktop/langchain/my_langchain_stuff/document_loaders/dl-curriculum.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)