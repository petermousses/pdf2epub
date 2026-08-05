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

## EPUB validation

Text-based PDFs use their embedded text layer so Unicode equation symbols are
preserved; scanned PDFs continue through OCR. Generated EPUBs package figure
references, split oversized content documents, and sanitize OCR markup before
the file is written.

For an existing library book, run the full validation endpoint:

```bash
curl -X POST 'http://localhost:8000/api/library/book.epub/validate-full?render=false'
```

The report checks EPUB structure, internal resources, rendering when
Chromium is available, and text/TOC/image fidelity when the source PDF is
available.
