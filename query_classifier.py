import os
import csv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv('QueryClassifier/API_KEY.env'))

# Initialize the chat model with the API key and model
chatmodel = ChatOpenAI(
    api_key=os.getenv('CHATGPT_API_KEY'),
    model='gpt-4'
)

# Read the base prompt from a file
with open('QueryClassifier/prompt.txt', 'r') as file:
    prompt_base = file.read().strip()

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["queries"],
    template=prompt_base
)

# Create the LLMChain
chain = LLMChain(llm=chatmodel, prompt=prompt_template)

# # Read the list of queries from a file
# with open('QueryClassifier/query_list.txt', 'r') as file:
#     queries = [query.strip() for query in file.readlines()]

with open('QueryClassifier/searchTerms-125 - Sheet1.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    queries = [row[0].strip() for row in reader]

# Function to divide queries into chunks
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Initialize a dictionary to count query types with predefined types
query_type_counts = {
    "Exact Search": 0,
    "Product Type Search": 0,
    "Symptom Search": 0,
    "Non-Product Search": 0,
    "Feature Search": 0,
    "Thematic Search": 0,
    "Relational Search": 0,
    "Compatibility Search": 0,
    "Subjective Search": 0,
    "Slang": 0,
    "Abbreviation": 0,
    "Symbol Search": 0,
    "Implicit Search": 0,
    "Natural Language Search": 0
}

all_batch_results = []  # List to store all batch results

# Process queries in batches of 10
batch_size = 10
batch_number = 0  # Initialize batch number counter
for query_batch in chunk_list(queries, batch_size):
    batch_number += 1  # Increment batch number
    print(f"Starting batch #{batch_number}")  # Print the current batch number
    combined_queries = "\n".join(query_batch)
    try:
        result = chain.invoke({"queries": combined_queries})
        query_types = result['text'].split(', ')
        # Zip the queries from the batch with their corresponding types
        batch_results = list(zip(query_batch, query_types))
        
        # Store results
        all_batch_results.extend(batch_results)
        
        # Count each query type
        for _, q_type in batch_results:
            if q_type in query_type_counts:
                query_type_counts[q_type] += 1
            else:
                print(f"Unexpected query type: {q_type}")  # Handle unexpected types
    except Exception as e:
        print(f"An error occurred: {e}")

# Write the original results to CSV
with open('QueryClassifier/results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Search Terms', 'Query Type'])  # Original header

    # Write the individual results
    for result in all_batch_results:
        writer.writerow([result[0], result[1]])  # Fill in the columns

# Write the query type counts to a new CSV file
with open('QueryClassifier/distribution.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Query Type', 'Count'])  # Header for distribution file

    # Write the counts
    for q_type, count in query_type_counts.items():
        writer.writerow([q_type, count])  # Write the query type and count

# # Initialize a dictionary to count query types with predefined types
# query_type_counts = {
#     "Exact Search": 0,
#     "Product Type Search": 0,
#     "Symptom Search": 0,
#     "Non-Product Search": 0,
#     "Feature Search": 0,
#     "Thematic Search": 0,
#     "Relational Search": 0,
#     "Compatibility Search": 0,
#     "Subjective Search": 0,
#     "Slang": 0,
#     "Abbreviation": 0,
#     "Symbol Search": 0,
#     "Implicit Search": 0,
#     "Natural Language Search": 0
# }

# all_batch_results = []  # List to store all batch results

# # Process queries in batches of 25
# batch_size = 25
# for query_batch in chunk_list(queries, batch_size):
#     combined_queries = "\n".join(query_batch)
#     try:
#         result = chain.invoke({"queries": combined_queries})
#         query_types = result['text'].split(', ')
#         # Zip the queries from the batch with their corresponding types
#         batch_results = list(zip(query_batch, query_types))
        
#         # Store results
#         all_batch_results.extend(batch_results)
        
#         # Count each query type
#         for _, q_type in batch_results:
#             if q_type in query_type_counts:
#                 query_type_counts[q_type] += 1
#             else:
#                 print(f"Unexpected query type: {q_type}")  # Handle unexpected types
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Write to CSV, counts first
# with open('QueryClassifier/results.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Search Terms', 'Query Type', 'Query Type (Baymar)', 'Count'])  # Write header

#     # Write the counts
#     for q_type, count in query_type_counts.items():
#         writer.writerow([None, None, q_type, count])  # Write the query type and count

#     # Write the individual results
#     for result in all_batch_results:
#         writer.writerow([result[0], result[1], None, None])  # Fill in the first two columns, leave last two blank
