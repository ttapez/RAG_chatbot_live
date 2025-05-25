import db, textwrap, sys

def main(name: str):
    db.init_db()
    t = db.create_tenant(name)
    print("Tenant created:\n", t)
    snippet = textwrap.dedent(f"""
        <script async
                src="https://18.133.103.39/static/widget.js"
                data-key="{t['api_key']}"></script>
    """).strip()
    print("\nEmbed snippet:\n", snippet)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_tenant.py \"Store Name\"")
        exit(1)
    main(sys.argv[1])
