---
id: cli_commands.md
summary: Interact with Milvus using commands.
title: Milvus_CLI Command Reference
---

# Milvus_CLI Command Reference

Milvus Command-Line Interface (CLI) is a command-line tool that supports database connection, data operations, and import and export of data.

This topic introduces all supported commands and the corresponding options. Some examples are also included for your reference.

## Command Groups

Milvus CLI commands are organized into the following groups:

- `create`: Create collection, database, partition, user, role, alias, index, privilege_group, or resource_group
- `delete`: Delete collection, database, partition, alias, user, role, index, entities, IDs, privilege_group, resource_group, connection_history, or collection_properties
- `list`: List collections, databases, partitions, users, roles, grants, indexes, aliases, connections, connection_history, privilege_groups, resource_groups, or bulk_insert_tasks
- `show`: Show collection, collection_stats, database, partition, partition_stats, index, index_progress, loading_progress, load_state, flush_state, compaction_state, compaction_plans, replicas, query_segment_info, role, user, alias, output, resource_group, or bulk_insert_state
- `grant`: Grant role, privilege, or privilege_group
- `revoke`: Revoke role, privilege, or privilege_group
- `load`: Load collection or partition
- `release`: Release collection or partition
- `use`: Use database
- `rename`: Rename collection
- `insert`: Insert entities (file or row)
- `upsert`: Upsert entities (file or row)
- `set`: Set output format
- `alter`: Alter database, collection_properties, or collection_field
- `update`: Update password or resource_group

## clear

Clears the screen.

<h3 id="clear">Syntax</h3>

```shell
clear
```

## connect

Connects to Milvus.

<h3 id="connect">Syntax</h3>

```shell
connect [-uri (text)] [-t (text)] [-tls (0|1|2)] [-cert (text)] [--save-as (text)]
```

<h3 id="connect">Options</h3>

| Option   | Full name | Description                                                                                                              |
| :------- | :-------- | :----------------------------------------------------------------------------------------------------------------------- |
| -uri     | --uri     | (Optional) The uri name. The default is "http://127.0.0.1:19530". Can also be set via `ZILLIZ_URI` environment variable. |
| -t       | --token   | (Optional) The zilliz cloud apikey or `username:password`. Can also be set via `ZILLIZ_TOKEN` environment variable.      |
| -tls     | --tlsmode | (Optional) Set TLS mode: 0 (No encryption), 1 (One-way encryption), 2 (Two-way encryption). Default is 0.               |
| -cert    | --cert    | (Optional) Path to the client certificate file. Works with one-way encryption.                                           |
| --save-as| n/a       | (Optional) Save connection with custom alias for later use.                                                              |
| --help   | n/a       | Displays help for using the command.                                                                                     |

<h3 id="connect">Examples</h3>

```shell
milvus_cli > connect -uri http://127.0.0.1:19530

milvus_cli > connect -uri http://192.168.1.100:19530 -t root:milvus

milvus_cli > connect -uri https://xxx.zillizcloud.com -t <api_key>
```

## disconnect

Disconnects from Milvus.

<h3 id="disconnect">Syntax</h3>

```shell
disconnect
```

## create database

Creates a database in Milvus.

<h3 id="create-database">Syntax</h3>

```shell
create database -db (text)
```

<h3 id="create-database">Options</h3>

| Option | Full name | Description                             |
| :----- | :-------- | :-------------------------------------- |
| -db    | --db_name | [Required] The database name in milvus. |
| --help | n/a       | Displays help for using the command.    |

<h3 id="create-database">Example</h3>

```shell
milvus_cli > create database -db testdb
```

## use database

Uses a database in Milvus.

<h3 id="use-database">Syntax</h3>

```shell
use database -db (text)
```

<h3 id="use-database">Options</h3>

| Option | Full name | Description                             |
| :----- | :-------- | :-------------------------------------- |
| -db    | --db_name | [Required] The database name in milvus. |
| --help | n/a       | Displays help for using the command.    |

<h3 id="use-database">Example</h3>

```shell
milvus_cli > use database -db testdb
```

## list databases

Lists all databases in Milvus.

<h3 id="list-database">Syntax</h3>

```shell
list databases
```

## show database

Shows details and properties of a database.

<h3 id="show-database">Syntax</h3>

```shell
show database [-db (text)]
```

<h3 id="show-database">Options</h3>

| Option | Full name | Description                                       |
| :----- | :-------- | :------------------------------------------------ |
| -db    | --db_name | (Optional) The database name. Defaults to current.|
| --help | n/a       | Displays help for using the command.               |

## alter database

Alters database properties.

<h3 id="alter-database">Syntax</h3>

```shell
alter database -db (text)
```

<h3 id="alter-database">Options</h3>

| Option | Full name | Description                             |
| :----- | :-------- | :-------------------------------------- |
| -db    | --db_name | [Required] The database name in milvus. |
| --help | n/a       | Displays help for using the command.    |

<h3 id="alter-database">Interactive Example</h3>

```shell
milvus_cli > alter database -db testdb

Property key: collection.ttl.seconds
Property value: 86400
```

## delete database

Deletes a database in Milvus.

<h3 id="delete-database">Syntax</h3>

```shell
delete database -db (text) [--yes]
```

<h3 id="delete-database">Options</h3>

| Option | Full name | Description                             |
| :----- | :-------- | :-------------------------------------- |
| -db    | --db_name | [Required] The database name in milvus. |
| --yes  | -y        | (Optional) Skip confirmation prompt.    |
| --help | n/a       | Displays help for using the command.    |

<h3 id="delete-database">Example</h3>

```shell
milvus_cli > delete database -db testdb

Warning! You are trying to delete the database. This action cannot be undone!
Do you want to continue? [y/N]: y
```

## create collection

Creates a collection.

<h3 id="create-collection">Syntax</h3>

```shell
create collection [--schema-file (text)]
```

<h3 id="create-collection">Options</h3>

| Option        | Full name     | Description                                     |
| :------------ | :------------ | :---------------------------------------------- |
| --schema-file | --schema-file | (Optional) Path to JSON file with schema definition. |
| --help        | n/a           | Displays help for using the command.             |

<h3 id="create-collection">Interactive Example</h3>

```shell
milvus_cli > create collection

Please input collection name: car
Please input auto id [False]: False
Please input description []: car collection
Is support dynamic field [False]: False
Please input consistency level(Strong(0),Bounded(1), Session(2), and Eventually(3)) [1]: 1
Please input shards number [1]: 1

Field name: id
Field type (INT64, VARCHAR, FLOAT_VECTOR, etc.): INT64
Field description []: primary key
Is id the primary key? [y/N]: y

Field name: vector
Field type (INT64, VARCHAR, FLOAT_VECTOR, etc.): FLOAT_VECTOR
Field description []: vector field
Dimension: 128

Field name:

Do you want to add embedding function? [y/N]: n
```

## list collections

Lists all collections in the current database.

<h3 id="list-collections">Syntax</h3>

```shell
list collections
```

## show collection

Shows the detailed information of a collection.

<h3 id="show-collection">Syntax</h3>

```shell
show collection -c (text)
```

<h3 id="show-collection">Options</h3>

| Option | Full name         | Description                          |
| :----- | :---------------- | :----------------------------------- |
| -c     | --collection-name | The name of the collection.          |
| --help | n/a               | Displays help for using the command. |

## show collection_stats

Shows collection statistics.

<h3 id="show-collection-stats">Syntax</h3>

```shell
show collection_stats -c (text)
```

<h3 id="show-collection-stats">Options</h3>

| Option | Full name         | Description                          |
| :----- | :---------------- | :----------------------------------- |
| -c     | --collection-name | [Required] The name of the collection.|
| --help | n/a               | Displays help for using the command. |

## rename collection

Renames a collection.

<h3 id="rename-collection">Syntax</h3>

```shell
rename collection -old (text) -new (text)
```

<h3 id="rename-collection">Options</h3>

| Option | Full name              | Description                          |
| :----- | :--------------------- | :----------------------------------- |
| -old   | --old-collection-name  | [Required] The old collection name.  |
| -new   | --new-collection-name  | [Required] The new collection name.  |
| --help | n/a                    | Displays help for using the command. |

## delete collection

Deletes a collection.

<h3 id="delete-collection">Syntax</h3>

```shell
delete collection -c (text) [--yes]
```

<h3 id="delete-collection">Options</h3>

| Option | Full name         | Description                               |
| :----- | :---------------- | :---------------------------------------- |
| -c     | --collection-name | [Required] The name of the collection.    |
| --yes  | -y                | (Optional) Skip confirmation prompt.      |
| --help | n/a               | Displays help for using the command.      |

<h3 id="delete-collection">Example</h3>

```shell
milvus_cli > delete collection -c car

Warning! You are trying to delete the collection. This action cannot be undone!
Do you want to continue? [y/N]: y
```

## load collection

Loads a collection into RAM.

<h3 id="load-collection">Syntax</h3>

```shell
load collection -c (text)
```

<h3 id="load-collection">Options</h3>

| Option | Full name         | Description                          |
| :----- | :---------------- | :----------------------------------- |
| -c     | --collection-name | The name of the collection.          |
| --help | n/a               | Displays help for using the command. |

## release collection

Releases a collection from RAM.

<h3 id="release-collection">Syntax</h3>

```shell
release collection -c (text)
```

<h3 id="release-collection">Options</h3>

| Option | Full name         | Description                          |
| :----- | :---------------- | :----------------------------------- |
| -c     | --collection-name | The name of the collection.          |
| --help | n/a               | Displays help for using the command. |

## truncate

Removes all data from a collection but keeps the schema.

<h3 id="truncate">Syntax</h3>

```shell
truncate -c (text) [--yes]
```

<h3 id="truncate">Options</h3>

| Option | Full name         | Description                            |
| :----- | :---------------- | :------------------------------------- |
| -c     | --collection-name | [Required] The name of the collection. |
| --yes  | -y                | (Optional) Skip confirmation prompt.   |
| --help | n/a               | Displays help for using the command.   |

<h3 id="truncate">Example</h3>

```shell
milvus_cli > truncate -c car

Warning!
You are trying to remove all data in the collection. This action cannot be undone!
Do you want to continue? [y/N]: y
```

## flush

Flushes collection data to storage.

<h3 id="flush">Syntax</h3>

```shell
flush -c (text) [-t (number)]
```

<h3 id="flush">Options</h3>

| Option | Full name         | Description                            |
| :----- | :---------------- | :------------------------------------- |
| -c     | --collection-name | [Required] The name of the collection. |
| -t     | --timeout         | (Optional) Timeout in seconds.         |
| --help | n/a               | Displays help for using the command.   |

## flush_all

Flushes all collections to storage.

<h3 id="flush-all">Syntax</h3>

```shell
flush_all [-t (number)]
```

<h3 id="flush-all">Options</h3>

| Option | Full name | Description                          |
| :----- | :-------- | :----------------------------------- |
| -t     | --timeout | (Optional) Timeout in seconds.       |
| --help | n/a       | Displays help for using the command. |

## show flush_state

Shows flush state for a collection.

<h3 id="show-flush-state">Syntax</h3>

```shell
show flush_state -c (text)
```

<h3 id="show-flush-state">Options</h3>

| Option | Full name         | Description                            |
| :----- | :---------------- | :------------------------------------- |
| -c     | --collection-name | [Required] The name of the collection. |
| --help | n/a               | Displays help for using the command.   |

## compact

Compacts a collection to merge small segments and remove deleted data.

<h3 id="compact">Syntax</h3>

```shell
compact -c (text) [-t (number)]
```

<h3 id="compact">Options</h3>

| Option | Full name         | Description                            |
| :----- | :---------------- | :------------------------------------- |
| -c     | --collection-name | [Required] The name of the collection. |
| -t     | --timeout         | (Optional) Timeout in seconds.         |
| --help | n/a               | Displays help for using the command.   |

## show compaction_state

Shows compaction state.

<h3 id="show-compaction-state">Syntax</h3>

```shell
show compaction_state -id (number)
```

<h3 id="show-compaction-state">Options</h3>

| Option | Full name       | Description                           |
| :----- | :-------------- | :------------------------------------ |
| -id    | --compaction-id | [Required] The compaction ID.         |
| --help | n/a             | Displays help for using the command.  |

## show compaction_plans

Shows compaction plans.

<h3 id="show-compaction-plans">Syntax</h3>

```shell
show compaction_plans -c (text) -id (number)
```

<h3 id="show-compaction-plans">Options</h3>

| Option | Full name         | Description                            |
| :----- | :---------------- | :------------------------------------- |
| -c     | --collection-name | [Required] The name of the collection. |
| -id    | --compaction-id   | [Required] The compaction ID.          |
| --help | n/a               | Displays help for using the command.   |

## show loading_progress

Displays the progress of loading a collection.

<h3 id="show-loading-progress">Syntax</h3>

```shell
show loading_progress -c (text)
```

<h3 id="show-loading-progress">Options</h3>

| Option | Full name         | Description                          |
| :----- | :---------------- | :----------------------------------- |
| -c     | --collection-name | The name of the collection.          |
| --help | n/a               | Displays help for using the command. |

## show load_state

Shows the load state of a collection or partition.

<h3 id="show-load-state">Syntax</h3>

```shell
show load_state -c (text) [-p (text)]
```

<h3 id="show-load-state">Options</h3>

| Option | Full name         | Description                            |
| :----- | :---------------- | :------------------------------------- |
| -c     | --collection-name | [Required] The name of the collection. |
| -p     | --partition       | (Optional) The name of the partition.  |
| --help | n/a               | Displays help for using the command.   |

## show replicas

Shows replicas information for a collection.

<h3 id="show-replicas">Syntax</h3>

```shell
show replicas -c (text)
```

<h3 id="show-replicas">Options</h3>

| Option | Full name         | Description                            |
| :----- | :---------------- | :------------------------------------- |
| -c     | --collection-name | [Required] The name of the collection. |
| --help | n/a               | Displays help for using the command.   |

## show query_segment_info

Shows query segment information for a collection.

<h3 id="show-query-segment-info">Syntax</h3>

```shell
show query_segment_info -c (text)
```

<h3 id="show-query-segment-info">Options</h3>

| Option | Full name         | Description                            |
| :----- | :---------------- | :------------------------------------- |
| -c     | --collection-name | [Required] The name of the collection. |
| --help | n/a               | Displays help for using the command.   |
