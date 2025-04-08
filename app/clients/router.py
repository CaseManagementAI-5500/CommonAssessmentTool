"""
Router module for client-related endpoints.
Handles all HTTP requests for client operations including create, read, update, and delete.
"""
from typing import List, Optional

from fastapi import HTTPException, APIRouter, Depends, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.router import get_current_user, get_admin_user
from app.database import get_db
from app.models import User
from app.clients.service.client_service import ClientService
from app.clients.schema import (
    ClientResponse,
    ClientUpdate,
    ClientListResponse,
    ServiceResponse,
    ServiceUpdate,
)
from app.clients.service.model_factory import (
    get_current_model_name,
    set_current_model,
    get_available_models,
)

from app.clients.service.logic import interpret_and_calculate
from app.clients.schema import PredictionInput

router = APIRouter(prefix="/clients", tags=["clients"])


@router.post("/predictions")
async def predict(data: PredictionInput):
    return interpret_and_calculate(data.model_dump())


@router.get("/models/available")
async def list_available_models(current_user: User = Depends(get_current_user)):
    """Retrieve the list of all available models"""
    return {"models": get_available_models(), "current_model": get_current_model_name()}


@router.get("/models/current")
async def get_current_model(current_user: User = Depends(get_current_user)):
    """Retrieve the current model name being used"""
    return {"name": get_current_model_name()}


@router.put("/models/switch/{model_name}")
async def switch_model(
    model_name: str,
    current_user: User = Depends(get_admin_user),
):
    """Switch to a different prediction model"""
    try:
        new_model = set_current_model(model_name)
        return {"name": new_model}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/", response_model=ClientListResponse)
async def get_clients(
    current_user: User = Depends(get_admin_user),  # Required for authorization
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        default=50, ge=1, le=150, description="Maximum number of records to return"
    ),
    db: Session = Depends(get_db),
) -> ClientListResponse:
    """
    Retrieve a paginated list of all clients.

    Args:
        current_user: Authenticated admin user making the request
        skip: Number of records to skip for pagination
        limit: Maximum number of records to return
        db: Database session

    Returns:
        ClientListResponse: Paginated list of clients
    """
    return ClientService.get_clients(db, skip, limit)


@router.get("/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: int,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
) -> ClientResponse:
    """
    Get a specific client by ID.

    Args:
        client_id: Unique identifier of the client
        current_user: Authenticated admin user making the request
        db: Database session

    Returns:
        ClientResponse: Client details
    """
    return ClientService.get_client(db, client_id)


# For the function with too many arguments, we can create a search parameters class


class ClientSearchCriteria(BaseModel):
    """Search criteria for filtering clients"""

    employment_status: Optional[bool] = None
    education_level: Optional[int] = None
    age_min: Optional[int] = None
    gender: Optional[int] = None
    work_experience: Optional[int] = None
    canada_workex: Optional[int] = None
    dep_num: Optional[int] = None
    canada_born: Optional[bool] = None
    citizen_status: Optional[bool] = None
    fluent_english: Optional[bool] = None
    reading_english_scale: Optional[int] = None
    speaking_english_scale: Optional[int] = None
    writing_english_scale: Optional[int] = None
    numeracy_scale: Optional[int] = None
    computer_scale: Optional[int] = None
    transportation_bool: Optional[bool] = None
    caregiver_bool: Optional[bool] = None
    housing: Optional[int] = None
    income_source: Optional[int] = None
    felony_bool: Optional[bool] = None
    attending_school: Optional[bool] = None
    substance_use: Optional[bool] = None
    time_unemployed: Optional[int] = None
    need_mental_health_support_bool: Optional[bool] = None


@router.get("/search/by-criteria", response_model=List[ClientResponse])
async def get_clients_by_criteria(
    search_criteria: ClientSearchCriteria = Depends(),
    current_user: User = Depends(get_admin_user),  # Required for authorization
    db: Session = Depends(get_db),
) -> List[ClientResponse]:
    """
    Search clients by any combination of criteria.

    Args:
        search_criteria: Search parameters for filtering clients
        current_user: Authenticated admin user making the request
        db: Database session

    Returns:
        List[ClientResponse]: List of matching clients
    """
    return ClientService.get_clients_by_criteria(db, **search_criteria.dict())


class ServiceSearchCriteria(BaseModel):
    """Search criteria for filtering clients by services"""

    employment_assistance: Optional[bool] = None
    life_stabilization: Optional[bool] = None
    retention_services: Optional[bool] = None
    specialized_services: Optional[bool] = None
    employment_related_financial_supports: Optional[bool] = None
    employer_financial_supports: Optional[bool] = None
    enhanced_referrals: Optional[bool] = None


@router.get("/search/by-services", response_model=List[ClientResponse])
async def get_clients_by_services(
    service_criteria: ServiceSearchCriteria = Depends(),
    current_user: User = Depends(get_admin_user),  # Required for authorization
    db: Session = Depends(get_db),
) -> List[ClientResponse]:
    """
    Get clients filtered by multiple service statuses.

    Args:
        service_criteria: Service-related search parameters
        current_user: Authenticated admin user making the request
        db: Database session

    Returns:
        List[ClientResponse]: List of matching clients
    """
    return ClientService.get_clients_by_services(db, **service_criteria.dict())


@router.get("/{client_id}/services", response_model=List[ServiceResponse])
async def get_client_services(
    client_id: int,
    current_user: User = Depends(get_admin_user),  # Required for authorization
    db: Session = Depends(get_db),
) -> List[ServiceResponse]:
    """
    Get all services and their status for a specific client.

    Args:
        client_id: Unique identifier of the client
        current_user: Authenticated admin user making the request
        db: Database session

    Returns:
        List[ServiceResponse]: List of client services
    """
    return ClientService.get_client_services(db, client_id)


@router.get("/search/success-rate", response_model=List[ClientResponse])
async def get_clients_by_success_rate(
    min_rate: int = Query(
        70, ge=0, le=100, description="Minimum success rate percentage"
    ),
    current_user: User = Depends(get_admin_user),  # Required for authorization
    db: Session = Depends(get_db),
) -> List[ClientResponse]:
    """
    Get clients with success rate above specified threshold.

    Args:
        min_rate: Minimum success rate percentage
        current_user: Authenticated admin user making the request
        db: Database session

    Returns:
        List[ClientResponse]: List of clients meeting success rate criteria
    """
    return ClientService.get_clients_by_success_rate(db, min_rate)


@router.get("/case-worker/{case_worker_id}", response_model=List[ClientResponse])
async def get_clients_by_case_worker(
    case_worker_id: int,
    current_user: User = Depends(get_current_user),  # Required for authorization
    db: Session = Depends(get_db),
) -> List[ClientResponse]:
    """
    Get all clients assigned to a specific case worker.

    Args:
        case_worker_id: ID of the case worker
        current_user: Authenticated user making the request
        db: Database session

    Returns:
        List[ClientResponse]: List of clients assigned to the case worker
    """
    return ClientService.get_clients_by_case_worker(db, case_worker_id)


@router.put("/{client_id}", response_model=ClientResponse)
async def update_client(
    client_id: int,
    client_data: ClientUpdate,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Update a client's information"""
    return ClientService.update_client(db, client_id, client_data)


@router.put("/{client_id}/services/{user_id}", response_model=ServiceResponse)
async def update_client_services(
    client_id: int,
    user_id: int,
    service_update: ServiceUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update the service status and details for a specific client-user pair.

    Args:
        client_id: Unique identifier of the client
        user_id: Unique identifier of the user (case worker)
        service_update: Updated service information containing status changes
            for various service types (employment assistance, life stabilization, etc.)
        current_user: Authenticated user making the request (used for authorization)
        db: Database session

    Returns:
        ServiceResponse: Updated service details including:
            - Employment assistance status
            - Life stabilization status
            - Retention services status
            - Specialized services status
            - Employment-related financial supports status
            - Employer financial supports status
            - Enhanced referrals status
            - Success rate (if applicable)

    Raises:
        HTTPException:
            - 404 if client or user not found
            - 403 if user doesn't have permission to update services
            - 400 if service update data is invalid
    """
    return ClientService.update_client_services(db, client_id, user_id, service_update)


@router.post("/{client_id}/case-assignment", response_model=ServiceResponse)
async def create_case_assignment(
    client_id: int,
    case_worker_id: int = Query(..., description="Case worker ID to assign"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Create a new case assignment for a client with a case worker"""
    return ClientService.create_case_assignment(db, client_id, case_worker_id)


@router.delete("/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_client(
    client_id: int,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Delete a client"""
    ClientService.delete_client(db, client_id)
    return None
