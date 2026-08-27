import json
from pathlib import Path

from pydantic import BaseModel


class FieldCatalogError(ValueError):
    """Raised when a field catalog contains invalid aliases."""


class FieldDefinition(BaseModel):
    key: str
    type: str = "text"
    aliases: list[str]


class FieldCatalogFile(BaseModel):
    version: str
    fields: list[FieldDefinition]


class FieldCatalog:
    def __init__(self, data: FieldCatalogFile) -> None:
        self.version = data.version
        self.fields = {field.key: field for field in data.fields}
        self.alias_to_canonical: dict[str, str] = {}
        for field in data.fields:
            for alias in [field.key, *field.aliases]:
                normalized = alias.strip()
                previous = self.alias_to_canonical.get(normalized)
                if previous is not None and previous != field.key:
                    raise FieldCatalogError(
                        f"duplicate alias {alias!r}: {previous}, {field.key}"
                    )
                self.alias_to_canonical[normalized] = field.key

    @classmethod
    def load(cls, path: Path) -> "FieldCatalog":
        data = FieldCatalogFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
        return cls(data)

    def resolve(self, raw_name: str) -> str | None:
        return self.alias_to_canonical.get(raw_name.strip())
