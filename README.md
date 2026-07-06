# pdf2epub

## EPUB validation

Converted EPUBs can be validated at three depths (`backend/app/epub_validator.py`):

1. **Format** — pure-Python OCF/OPF/content checks (container structure,
   manifest↔archive cross-checks, spine resolution, broken internal
   references, image decodability) plus the official W3C **EPUBCheck**
   validator when Java is available.
2. **Render simulation** — every spine document is opened in headless
   Chromium (Playwright) the way a Blink/WebKit-based reader would render
   it, catching broken images, failed resource loads, script/page errors,
   blank chapters, and horizontal overflow. Screenshots are captured for
   visual inspection.
3. **PDF fidelity** — when the source PDF is supplied, the EPUB text is
   compared against it with word n-gram containment: overall text coverage,
   the exact PDF page ranges that went missing, extra/hallucinated content,
   PDF bookmark (TOC) coverage, image counts, and mojibake detection.

Layers degrade gracefully: missing Java/EPUBCheck or Chromium marks that
layer "skipped" instead of failing the run.

### API

```
POST /api/library/{filename}/validate-full    # multipart field 'pdf' optional
GET  /api/library/{filename}/validation-report
GET  /api/library/{filename}/screenshots/{shot}.png
```

### CLI

```bash
cd backend
python3 -m app.epub_validator book.epub source.pdf \
    --screenshots shots/ --json report.json
```

Exit code is 0 only when the verdict is `pass`.

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