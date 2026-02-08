"""
Data Lake MCP Server

Provides data access tools:
- query_datasets: Search available datasets
- list_slides: List slides in a dataset
- get_cohort: Get patient cohort data
"""

import random
from datetime import datetime, timedelta
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Data Lake MCP Server")


# Mock datasets
MOCK_DATASETS = {
    "tcga-brca": {
        "id": "tcga-brca",
        "name": "TCGA Breast Cancer",
        "description": "The Cancer Genome Atlas Breast Invasive Carcinoma collection",
        "samples": 1098,
        "slides": 2196,
        "data_types": ["WSI", "clinical", "genomic", "transcriptomic"],
        "cancer_type": "Breast",
        "access_level": "public",
        "last_updated": "2024-01-15"
    },
    "tcga-luad": {
        "id": "tcga-luad",
        "name": "TCGA Lung Adenocarcinoma",
        "description": "The Cancer Genome Atlas Lung Adenocarcinoma collection",
        "samples": 585,
        "slides": 1170,
        "data_types": ["WSI", "clinical", "genomic"],
        "cancer_type": "Lung",
        "access_level": "public",
        "last_updated": "2024-01-10"
    },
    "internal-breast-2024": {
        "id": "internal-breast-2024",
        "name": "MCP Breast Cancer Cohort 2024",
        "description": "Proprietary breast cancer cohort with detailed annotations",
        "samples": 2500,
        "slides": 5000,
        "data_types": ["WSI", "clinical", "treatment_response"],
        "cancer_type": "Breast",
        "access_level": "internal",
        "last_updated": "2024-02-01"
    },
    "pharma-trial-001": {
        "id": "pharma-trial-001",
        "name": "PharmaCorp Immunotherapy Trial",
        "description": "Clinical trial data for checkpoint inhibitor study",
        "samples": 450,
        "slides": 900,
        "data_types": ["WSI", "clinical", "biomarkers", "response"],
        "cancer_type": "Multiple",
        "access_level": "restricted",
        "last_updated": "2024-01-28"
    }
}

MOCK_COHORTS = {
    "responders-io": {
        "id": "responders-io",
        "name": "Immunotherapy Responders",
        "description": "Patients who responded to checkpoint inhibitor therapy",
        "n_patients": 156,
        "criteria": "CR or PR at 6 months",
        "source_datasets": ["pharma-trial-001"]
    },
    "her2-positive": {
        "id": "her2-positive",
        "name": "HER2+ Breast Cancer",
        "description": "HER2 positive breast cancer patients",
        "n_patients": 342,
        "criteria": "IHC 3+ or FISH amplified",
        "source_datasets": ["tcga-brca", "internal-breast-2024"]
    },
    "early-stage-nsclc": {
        "id": "early-stage-nsclc",
        "name": "Early Stage NSCLC",
        "description": "Stage I-II non-small cell lung cancer",
        "n_patients": 289,
        "criteria": "TNM Stage I or II",
        "source_datasets": ["tcga-luad"]
    }
}


class ToolCallRequest(BaseModel):
    tool: str
    arguments: dict


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "data-lake"}


@app.get("/tools")
async def list_tools():
    """List available tools (MCP discovery)."""
    return {
        "tools": [
            {
                "name": "query_datasets",
                "description": "Search available datasets by cancer type, data types, or keywords. Returns matching datasets with metadata.",
                "parameters": {
                    "query": {"type": "string", "description": "Search query (e.g., 'breast cancer', 'lung', 'immunotherapy')"},
                    "filters": {"type": "object", "description": "Optional filters: cancer_type, data_types, access_level, min_samples"}
                }
            },
            {
                "name": "list_slides",
                "description": "List slides in a specific dataset with optional pagination.",
                "parameters": {
                    "dataset_id": {"type": "string", "description": "Dataset identifier"},
                    "limit": {"type": "integer", "description": "Maximum number of slides to return (default 10)"}
                }
            },
            {
                "name": "get_cohort",
                "description": "Get detailed information about a patient cohort including demographics and clinical characteristics.",
                "parameters": {
                    "cohort_id": {"type": "string", "description": "Cohort identifier"}
                }
            }
        ]
    }


@app.post("/call")
async def call_tool(request: ToolCallRequest):
    """Handle tool calls (MCP invocation)."""
    tool = request.tool
    args = request.arguments

    if tool == "query_datasets":
        return await query_datasets(args.get("query", ""), args.get("filters", {}))
    elif tool == "list_slides":
        return await list_slides(args.get("dataset_id"), args.get("limit", 10))
    elif tool == "get_cohort":
        return await get_cohort(args.get("cohort_id"))
    else:
        return {"error": f"Unknown tool: {tool}"}


async def query_datasets(query: str, filters: dict) -> dict:
    """Search available datasets."""
    query_lower = query.lower()
    results = []

    for dataset_id, dataset in MOCK_DATASETS.items():
        # Simple search matching
        match = False
        if not query:
            match = True
        elif query_lower in dataset["name"].lower():
            match = True
        elif query_lower in dataset["description"].lower():
            match = True
        elif query_lower in dataset["cancer_type"].lower():
            match = True

        # Apply filters
        if match and filters:
            if "cancer_type" in filters and filters["cancer_type"].lower() != dataset["cancer_type"].lower():
                match = False
            if "access_level" in filters and filters["access_level"] != dataset["access_level"]:
                match = False
            if "min_samples" in filters and dataset["samples"] < filters["min_samples"]:
                match = False
            if "data_types" in filters:
                required_types = filters["data_types"] if isinstance(filters["data_types"], list) else [filters["data_types"]]
                if not all(dt in dataset["data_types"] for dt in required_types):
                    match = False

        if match:
            results.append({
                **dataset,
                "relevance_score": round(random.uniform(0.7, 1.0), 2) if query else 1.0
            })

    # Sort by relevance
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    return {
        "query": query,
        "filters": filters,
        "total_results": len(results),
        "datasets": results,
        "available_filters": {
            "cancer_types": list(set(d["cancer_type"] for d in MOCK_DATASETS.values())),
            "access_levels": ["public", "internal", "restricted"],
            "data_types": ["WSI", "clinical", "genomic", "transcriptomic", "biomarkers", "response"]
        },
        "queried_at": datetime.utcnow().isoformat()
    }


async def list_slides(dataset_id: str, limit: int = 10) -> dict:
    """List slides in a dataset."""
    if dataset_id not in MOCK_DATASETS:
        return {"error": f"Dataset {dataset_id} not found", "available_datasets": list(MOCK_DATASETS.keys())}

    dataset = MOCK_DATASETS[dataset_id]
    limit = min(limit, 50)  # Cap at 50

    # Generate mock slide list
    slides = []
    for i in range(limit):
        slide_id = f"S-{dataset_id.upper()}-{i+1:05d}"
        slides.append({
            "slide_id": slide_id,
            "patient_id": f"P-{random.randint(10000, 99999)}",
            "tissue_type": dataset["cancer_type"],
            "stain": random.choice(["H&E", "IHC-PD-L1", "IHC-Ki67", "IHC-CD8"]),
            "diagnosis": random.choice(["Invasive carcinoma", "In situ carcinoma", "Normal adjacent", "Metastasis"]),
            "quality_score": round(random.uniform(0.7, 0.99), 2),
            "scan_date": (datetime.utcnow() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
            "has_annotations": random.choice([True, False]),
            "dimensions": {
                "width": random.randint(50000, 100000),
                "height": random.randint(40000, 80000)
            }
        })

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset["name"],
        "total_slides": dataset["slides"],
        "returned": len(slides),
        "limit": limit,
        "slides": slides,
        "has_more": dataset["slides"] > limit,
        "pagination": {
            "offset": 0,
            "next_offset": limit if dataset["slides"] > limit else None
        }
    }


async def get_cohort(cohort_id: str) -> dict:
    """Get patient cohort data."""
    if cohort_id not in MOCK_COHORTS:
        return {"error": f"Cohort {cohort_id} not found", "available_cohorts": list(MOCK_COHORTS.keys())}

    cohort = MOCK_COHORTS[cohort_id]
    n = cohort["n_patients"]

    # Generate mock demographics
    return {
        "cohort_id": cohort_id,
        "cohort_name": cohort["name"],
        "description": cohort["description"],
        "selection_criteria": cohort["criteria"],
        "source_datasets": cohort["source_datasets"],
        "n_patients": n,
        "demographics": {
            "age": {
                "mean": round(random.uniform(55, 68), 1),
                "std": round(random.uniform(10, 15), 1),
                "min": random.randint(25, 40),
                "max": random.randint(75, 90)
            },
            "sex": {
                "female": int(n * random.uniform(0.4, 0.7)),
                "male": int(n * random.uniform(0.3, 0.6))
            },
            "ethnicity": {
                "white": int(n * 0.65),
                "black": int(n * 0.15),
                "asian": int(n * 0.12),
                "other": int(n * 0.08)
            }
        },
        "clinical_characteristics": {
            "stage_distribution": {
                "I": int(n * random.uniform(0.15, 0.25)),
                "II": int(n * random.uniform(0.25, 0.35)),
                "III": int(n * random.uniform(0.25, 0.35)),
                "IV": int(n * random.uniform(0.1, 0.2))
            },
            "grade_distribution": {
                "1": int(n * random.uniform(0.1, 0.2)),
                "2": int(n * random.uniform(0.35, 0.45)),
                "3": int(n * random.uniform(0.35, 0.45))
            },
            "treatment_received": {
                "surgery": int(n * 0.85),
                "chemotherapy": int(n * 0.65),
                "radiotherapy": int(n * 0.45),
                "immunotherapy": int(n * 0.25)
            }
        },
        "outcomes": {
            "median_follow_up_months": round(random.uniform(24, 48), 1),
            "overall_survival_rate_5yr": round(random.uniform(0.6, 0.85), 2),
            "progression_free_survival_rate_2yr": round(random.uniform(0.5, 0.75), 2),
            "response_rate": round(random.uniform(0.3, 0.6), 2) if "response" in str(cohort.get("criteria", "")) else None
        },
        "data_availability": {
            "wsi_available": n,
            "clinical_complete": int(n * 0.95),
            "genomic_available": int(n * random.uniform(0.3, 0.7)),
            "biomarkers_available": int(n * random.uniform(0.5, 0.8))
        },
        "last_updated": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
