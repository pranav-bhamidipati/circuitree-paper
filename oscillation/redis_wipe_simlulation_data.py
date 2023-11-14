import redis

from oscillation_app import app


def main(
    database_url: str = None,
    table_name: str = "parameter_table",
    progress: bool = False,
):
    print("Connecting to database")
    database_url = database_url or app.conf["broker_url"]
    database: redis.Redis = redis.Redis.from_url(database_url)

    # Store in database
    clear_before_storing = (
        input(
            "Are you sure you want to delete ALL data from simulations? [y/N] "
        ).lower()
        == "y"
    )
    if not clear_before_storing:
        print("Not clearing database. Exiting.")
        return

    print(f"Deleting all keys except the parameter table {table_name}...")
    iterator: list = database.keys()
    n_keys = len(iterator)
    iterator = (i for i in iterator if i != table_name)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, total=n_keys - 1, desc="Deleting keys")
    for key in iterator:
        database.delete(key)

    print("Done.")


if __name__ == "__main__":
    main(progress=True)
