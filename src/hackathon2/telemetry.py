"""Azure Monitor / Application Insights via OpenTelemetry (handout section 11).

When APPLICATIONINSIGHTS_CONNECTION_STRING is set (the Bicep template injects it
into the Container App), requests, dependencies, logs and exceptions flow to
Application Insights. Without it this is a no-op, so local runs and tests need
nothing extra.

For agent-level traces, wrap workflow steps in spans:

    from hackathon2.telemetry import tracer
    with tracer.start_as_current_span("specialist.security") as span:
        span.set_attribute("vendor", request.vendor_name)
"""

import logging

from opentelemetry import trace

from hackathon2.config import Settings

SERVICE_NAME = "hackathon2"

tracer = trace.get_tracer(SERVICE_NAME)
_configured = False


def configure_telemetry(settings: Settings) -> bool:
    """Wire up Azure Monitor once per process. Returns True when telemetry is on.

    Must run BEFORE the FastAPI app is created: the FastAPI instrumentation
    patches the class, so only apps built afterwards are traced.
    """
    global _configured
    if _configured:
        return True
    if not settings.applicationinsights_connection_string:
        return False

    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry.sdk.resources import Resource

    configure_azure_monitor(
        connection_string=settings.applicationinsights_connection_string,
        logger_name=SERVICE_NAME,
        resource=Resource.create(
            {
                "service.name": SERVICE_NAME,
                "service.version": settings.image_tag,
                "deployment.environment": settings.app_env,
            }
        ),
    )
    logging.getLogger(SERVICE_NAME).info("Azure Monitor telemetry enabled")
    _configured = True
    return True
