from metadata_loader import MetadataLoader


loader = MetadataLoader("metadata.csv")

print("========== TABLES ==========")

print(loader.get_all_tables())

print()

print("========== COLUMNS ==========")

print(loader.get_all_columns())