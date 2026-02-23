# Boilerplates

Initial setups for FastAPI, NodeJS, Java, DotNet, Flutter and ReactJS (Under Development, and the readme file not completed, validated, or verified)

## Table of content

- [Setups](#setups)
  - [Table of content](#table-of-content)
  - [Development](#development)
    - [Prepatre the Templates](#Prepatre-the-Templates)
    - [Prerequisites](#prerequisites)
    - [Configuration](#configuration)
    - [Run instructions](#run-instructions)
    - [Helpful tools for development](#helpful-tools-for-development)
  - [Docker instructions](#docker-instructions)
    - [For development](#for-development)
    - [For production](#for-production)
    - [Helpful commands](#helpful-commands)
  - [Architecture](#architecture)
  - [TODOs](#todos)
  - [CONTRIBUTION](#contribution)

## Development
### Prepatre the Templates
- Select requoired Templates
- Remove the rest of the templates
- Ensure the prerequiresites[See Prerequisites section](#Prerequisites)
- Prepare the configureations[See configuration section](#configuration)
- Prepare docker compose file[See Docker instructions section](#Docker-instructions)

### Prerequisites
#### For NodeJs
- Node.js v16.13.1
- MongoDB v5.0.8
- Yarn v1.22.15 or higher
#### For FastAPI API: Without DB connection
#### For FastAPI Backend: With DB Connection
#### For Dotnet
#### For SpringBot
#### For ReactJS
#### For Flutter
#### For ReactJS


### Configuration
- Copy example.dev.env to dev.env 
- Select the require vatrable based on selected template and databses
- Adapt you variables if needed as

- Database (MongoDB)

```env
MONGODB_USER=${USER}
MONGODB_PASSWORD=${PASSWORD}
MONGODB_DATABASE=${DATABASE}
MONGODB_LOCAL_PORT=${LOCAL_PORT}
MONGODB_DOCKER_PORT=${DOCKER_PORT}
```

- Ports

```env
NODE_LOCAL_PORT=${NODE_LOCAL_PORT}
NODE_DOCKER_PORT=${NODE_DOCKER_PORT}
REACT_LOCAL_PORT=${REACT_LOCAL_PORT}
```

- Backend URL

```env
REACT_APP_BACKEND_BASE_URL=${URL}
```

example (Windows) : `REACT_APP_BACKEND_BASE_URL=http://localhost:${NODE_LOCAL_PORT}`

example (Linux) : `REACT_APP_BACKEND_BASE_URL=http://0.0.0.0:${NODE_LOCAL_PORT}`

### Run instructions
#### For NodeJS
- Install node server dependencies

```sh
cd server
yarn
```

- You need to load env variables for the server from dev.env

```sh
# For windows (use Gitbash)
set -a && source ../dev.env && set +a
# Or
export $(grep -v '^#' ../dev.env | xargs)

# For linux
source ../dev.env

```

- Run node server

```sh
yarn start
```
#### For ReactJS

- Install react client dependencies

```sh
cd client
yarn
```

- You need to load env variables for the client from dev.env

```sh
# For windows (use Gitbash)
set -a && source ../dev.env && set +a
# Or
export $(grep -v '^#' ../dev.env | xargs)

# For linux
source ../dev.env

```

- Run react client

```sh
yarn start
```

### Helpful tools for development

- `Visual studio code` (VS)
- `Markdownlint` VS extension
- `ESLint` VS extension

## Docker instructions

### For development

- Select required templates and remove the rest
- Copy example.dev.env to dev.env and adapt you variables [See configuration section](#configuration)


- To run docker-compose file

```sh
docker-compose -f docker-compose.dev.yml --env-file example.dev.env up -d
```

### For production

- Copy example.env to .env and adapt you variables [See configuration section](#configuration)

- Build and Publish server image

```sh
cd server
make build
make publish
```

- Build and Publish client image

```sh
cd client
make build
make publish
```

- Test it

```sh
docker-compose up -d
```

### Helpful commands

- To connect to it server's or client's terminal

```sh
docker-compose -f docker-compose.dev.yml --env-file dev.env exec [server|client] /bin/bash
```

- Verify logs file for the server or client

```sh
docker-compose -f docker-compose.dev.yml --env-file dev.env logs [server|client]
```

## Architecture

- [Database-architecture](./docs/database_arch.drawio)
- [General-architecture](./docs/general_arch.drawio)
- [Detailed-architecture](./docs/detailsed_arch.drawio)

## TODOs

- [TODOs](./docs/TODOs.md)

## CONTRIBUTION

- [CONTRIBUTION](./docs/CONTRIBUTION.md)