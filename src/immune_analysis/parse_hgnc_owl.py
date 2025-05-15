import rdflib
import pandas as pd

# Load the HGNC OWL file
g = rdflib.Graph()
owl_file = "/project/orien/data/aws/24PRJ217UVA_IORIG/codes/data/processed/hgnc.owl"
print(f"Parsing OWL file: {owl_file}...")
try:
    g.parse(owl_file)
    print("Parsing complete.")
except Exception as e:
    print(f"Error parsing OWL file: {e}")
    exit()


# Define necessary namespaces based on the OWL snippet
RDFS = rdflib.Namespace("http://www.w3.org/2000/01/rdf-schema#")
OBOINOWL = rdflib.Namespace("http://www.geneontology.org/formats/oboInOwl#")

# SPARQL query using the correct properties and extracting the Ensembl ID
query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>

SELECT ?symbol ?ensembl_id
WHERE {
  # Find a gene node that has an rdfs:label (symbol)
  ?gene rdfs:label ?symbol .

  # Find a database cross-reference (DbXref) associated with that gene
  ?gene oboInOwl:hasDbXref ?ensembl_xref_string .

  # Filter to keep only the DbXrefs that start with "ENSEMBL:"
  FILTER(STRSTARTS(STR(?ensembl_xref_string), "ENSEMBL:"))

  # Extract the Ensembl ID by removing the "ENSEMBL:" prefix using REPLACE
  BIND(REPLACE(STR(?ensembl_xref_string), "^ENSEMBL:", "") AS ?ensembl_id)
}
"""

print("Executing SPARQL query...")
# Execute the query
results = g.query(query)

# Create a dictionary mapping Ensembl IDs to HGNC symbols
enum_results = list(results) # Convert generator to list to check length
print(f"Found {len(enum_results)} mappings in OWL file.")

enum_results_clean = [(str(row.ensembl_id), str(row.symbol)) for row in enum_results if row.ensembl_id and row.symbol]
print(f"Found {len(enum_results_clean)} valid (non-empty) Ensembl/Symbol pairs.")

ensembl_to_hugo = dict(enum_results_clean)
print(f"Created dictionary with {len(ensembl_to_hugo)} unique Ensembl ID keys.")

# Save the mapping to a CSV file
if ensembl_to_hugo:
    mapping_df = pd.DataFrame(list(ensembl_to_hugo.items()), columns=["Ensembl_ID", "HGNC_Symbol"])
    # Ensure no duplicate Ensembl IDs remain after dict conversion (shouldn't happen, but check)
    # mapping_df = mapping_df.drop_duplicates(subset="Ensembl_ID", keep="first") # Not needed after dict conversion
    output_path = "/project/orien/data/aws/24PRJ217UVA_IORIG/codes/output/ensembl_to_hugo.csv"
    mapping_df.to_csv(output_path, index=False)
    print(f"Mapping saved to {output_path} with {len(mapping_df)} unique Ensembl IDs.")
else:
    print("No mappings found or dictionary creation failed. The CSV file will not be created/updated.") 