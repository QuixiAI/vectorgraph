CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS http;

LOAD 'age';
SET search_path = ag_catalog, "$user", public;

CREATE OR REPLACE FUNCTION get_embedding(text_content TEXT)
RETURNS vector(768) AS $$
DECLARE
    service_url TEXT;
    response http_response;
    request_body TEXT;
    embedding_array FLOAT[];
    embedding_json JSONB;
    v_content_hash TEXT;
    cached_embedding vector(768);
BEGIN
    -- Generate hash for caching
    v_content_hash := encode(sha256(text_content::bytea), 'hex');

    -- Check cache first
    SELECT ec.embedding INTO cached_embedding
    FROM embedding_cache ec
    WHERE ec.content_hash = v_content_hash;

    IF FOUND THEN
        RETURN cached_embedding;
    END IF;

    -- Get service URL
    SELECT value INTO service_url FROM embedding_config WHERE key = 'service_url';

    -- Prepare request body
    request_body := json_build_object('inputs', text_content)::TEXT;

    -- Make HTTP request
    SELECT * INTO response FROM http_post(
        service_url,
        request_body,
        'application/json'
    );

    -- Check response status
    IF response.status != 200 THEN
        RAISE EXCEPTION 'Embedding service error: % - %', response.status, response.content;
    END IF;

    -- Parse response
    embedding_json := response.content::JSONB;

    -- Extract embedding array (handle different response formats)
    IF embedding_json ? 'embeddings' THEN
        -- Format: {"embeddings": [[...]]}
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text((embedding_json->'embeddings')->0)::FLOAT
        );
    ELSIF embedding_json ? 'embedding' THEN
        -- Format: {"embedding": [...]}
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text(embedding_json->'embedding')::FLOAT
        );
    ELSIF embedding_json ? 'data' THEN
        -- OpenAI format: {"data": [{"embedding": [...]}]}
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text((embedding_json->'data')->0->'embedding')::FLOAT
        );
    ELSIF jsonb_typeof(embedding_json->0) = 'array' THEN
        -- HuggingFace TEI format: [[...]] (array of arrays)
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text(embedding_json->0)::FLOAT
        );
    ELSE
        -- Flat array format: [...]
        embedding_array := ARRAY(
            SELECT jsonb_array_elements_text(embedding_json)::FLOAT
        );
    END IF;

    -- Validate embedding size
    IF array_length(embedding_array, 1) != 768 THEN
        RAISE EXCEPTION 'Invalid embedding dimension: expected 768, got %', array_length(embedding_array, 1);
    END IF;

    -- Cache the result
    INSERT INTO embedding_cache (content_hash, embedding)
    VALUES (v_content_hash, embedding_array::vector(768))
    ON CONFLICT DO NOTHING;

    RETURN embedding_array::vector(768);
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Failed to get embedding: %', SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Check embedding service health
CREATE OR REPLACE FUNCTION check_embedding_service_health()
RETURNS BOOLEAN AS $$
DECLARE
    service_url TEXT;
    health_url TEXT;
    response http_response;
BEGIN
    SELECT value INTO service_url FROM embedding_config WHERE key = 'service_url';

    -- Extract base URL (scheme + host + port) using regexp, then append /health
    -- e.g., http://embeddings:80/embed -> http://embeddings:80/health
    health_url := regexp_replace(service_url, '^(https?://[^/]+).*$', '\1/health');

    SELECT * INTO response FROM http_get(health_url);

    RETURN response.status = 200;
EXCEPTION
    WHEN OTHERS THEN
        RETURN FALSE;
END;
$$ LANGUAGE plpgsql;
