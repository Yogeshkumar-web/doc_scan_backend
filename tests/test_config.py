from app.config import Settings


def test_cors_origins_accepts_comma_separated_env_value():
    settings = Settings(cors_origins="https://frontend.example.com, https://admin.example.com")

    assert settings.cors_origin_list == [
        "https://frontend.example.com",
        "https://admin.example.com",
    ]


def test_cors_origins_accepts_json_array_env_value():
    settings = Settings(cors_origins='["https://frontend.example.com","https://admin.example.com"]')

    assert settings.cors_origin_list == [
        "https://frontend.example.com",
        "https://admin.example.com",
    ]
