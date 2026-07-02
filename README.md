# pdf2epub

## Traefik

The frontend container joins an external `proxy` docker network and carries
Traefik routing labels, so it can sit behind an existing Traefik reverse
proxy. This is optional — the container still works standalone via the
`3001:80` port mapping if you don't use Traefik.

1. Copy the environment file and set your domain (only necessary if you're
   already running Traefik with an external `proxy` network on this host):

   ```bash
   cp .env.example .env
   ```

2. Make sure the external network exists (created once, shared by all
   Traefik-fronted stacks):

   ```bash
   docker network create proxy
   ```

3. Start the stack as usual:

   ```bash
   docker compose up -d
   ```