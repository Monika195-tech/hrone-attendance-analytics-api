
import os
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()

client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    serverSelectionTimeoutMS=3000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0")


# --------------------------------------------------------------------------- #
# Time and calculation helpers
# --------------------------------------------------------------------------- #
def normalize_utc(dt: datetime) -> datetime:
    """MongoDB naive datetimes represent UTC; truncate to whole seconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).replace(microsecond=0)


def from_epoch_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC).replace(microsecond=0)


def to_epoch_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    return int(normalize_utc(dt).timestamp() * 1000)


def to_ist(dt: datetime) -> datetime:
    return normalize_utc(dt).astimezone(IST)


def attendance_date(punch_in: datetime, shift_start: str,
                    shift_end: str) -> str:
    local_in = to_ist(punch_in)
    start = tuple(map(int, shift_start.split(":")))
    end = tuple(map(int, shift_end.split(":")))

    if end <= start and (local_in.hour, local_in.minute) < end:
        local_in -= timedelta(days=1)

    return local_in.date().isoformat()


def compute_late_minutes(punch_in: datetime, shift_start: str,
                         shift_end: str) -> int:
    local_in = to_ist(punch_in)
    start = tuple(map(int, shift_start.split(":")))
    end = tuple(map(int, shift_end.split(":")))

    shift_date = local_in.date()
    if end <= start and (local_in.hour, local_in.minute) < end:
        shift_date -= timedelta(days=1)

    start_dt = datetime.combine(
        shift_date,
        datetime.strptime(shift_start, "%H:%M").time(),
        tzinfo=IST,
    )
    seconds_late = int((local_in - start_dt).total_seconds())

    return seconds_late // 60 if seconds_late > 600 else 0


def compute_work_hours(punch_in: datetime,
                       punch_out: datetime) -> float:
    seconds = int(
        (normalize_utc(punch_out) - normalize_utc(punch_in)).total_seconds()
    )
    hours = (Decimal(seconds) / Decimal(3600)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return float(hours)


def compute_overtime(punch_out: datetime, shift_start: str,
                     shift_end: str, date_str: str) -> int:
    start = tuple(map(int, shift_start.split(":")))
    end = tuple(map(int, shift_end.split(":")))
    shift_date = date.fromisoformat(date_str)

    if end <= start:
        shift_date += timedelta(days=1)

    shift_end_dt = datetime.combine(
        shift_date,
        datetime.strptime(shift_end, "%H:%M").time(),
        tzinfo=IST,
    )
    seconds_late = int((to_ist(punch_out) - shift_end_dt).total_seconds())
    minutes = max(0, seconds_late // 60)

    return minutes if minutes >= 30 else 0


def attendance_response(doc: dict) -> dict:
    """Return only the fields permitted by the attendance API contract."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": to_epoch_ms(doc.get("punch_in")),
        "punch_out": to_epoch_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0),
        "overtime_minutes": doc.get("overtime_minutes", 0),
        "half_day": doc.get("half_day", False),
        "history": [
            {
                "at": to_epoch_ms(entry["at"]),
                "by": entry["by"],
                "reason": entry["reason"],
                "changes": {
                    key: {
                        "from": (
                            to_epoch_ms(change["from"])
                            if key in ("punch_in", "punch_out")
                            else change.get("from")
                        ),
                        "to": (
                            to_epoch_ms(change["to"])
                            if key in ("punch_in", "punch_out")
                            else change.get("to")
                        ),
                    }
                    for key, change in entry.get("changes", {}).items()
                },
            }
            for entry in doc.get("history", [])
        ],
    }

# API endpoints

from pydantic import EmailStr
import re


class EmployeeCreate(BaseModel):
    emp_code: str
    name: str
    email: str
    department: str
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: date


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None
    status: str = "PRESENT"


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None


@app.on_event("startup")
def create_indexes():
    try:
        db.employees.create_index(
            [("emp_code", ASCENDING)], unique=True
        )
        db.attendance_logs.create_index(
            [("emp_code", ASCENDING), ("date", ASCENDING)],
            unique=True,
        )
        db.attendance_logs.create_index(
            [("date", DESCENDING), ("emp_code", ASCENDING)]
        )
    except PyMongoError:
        pass


@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except PyMongoError:
        raise HTTPException(
            status_code=503, detail="Database unavailable"
        )


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeCreate):
    if not re.fullmatch(r"EMP\d{4,6}", body.emp_code):
        raise HTTPException(422, "Invalid employee code")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", body.email):
        raise HTTPException(422, "Invalid email")
    if body.shift_start == body.shift_end:
        raise HTTPException(422, "Shift times must differ")

    doc = body.model_dump()
    doc["joined_on"] = body.joined_on.isoformat()
    doc["created_at"] = datetime.now(UTC).replace(microsecond=0)

    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "Employee already exists")
    except PyMongoError:
        raise HTTPException(503, "Database unavailable")

    return {
        **{k: v for k, v in doc.items() if k != "_id"},
        "created_at": to_epoch_ms(doc["created_at"]),
    }


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    filt = {"department": department} if department else {}
    try:
        total = db.employees.count_documents(filt)
        docs = list(
            db.employees.find(filt, {"_id": 0})
            .sort("emp_code", ASCENDING)
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
    except PyMongoError:
        raise HTTPException(503, "Database unavailable")

    for d in docs:
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = to_epoch_ms(d["created_at"])
    return {"items": docs, "total": total, "page": page,
            "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "Employee not found")
    if body.status not in ("PRESENT", "WFH", "ON_DUTY"):
        raise HTTPException(422, "Invalid punch-in status")

    now = from_epoch_ms(body.punched_at) if body.punched_at is not None else datetime.now(UTC).replace(microsecond=0)
    if body.punched_at is not None and (
        not isinstance(body.punched_at, int)
        or isinstance(body.punched_at, bool)
        or body.punched_at < 100000000000
        or body.punched_at > 4102444800000
    ):
        raise HTTPException(422, "Invalid timestamp")

    day = attendance_date(now, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": day,
        "status": body.status,
        "punch_in": now,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(
            now, emp["shift_start"], emp["shift_end"]
        ),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "Attendance already exists")
    except PyMongoError:
        raise HTTPException(503, "Database unavailable")
    return attendance_response(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutRequest):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(404, "Employee not found")

    now = from_epoch_ms(body.punched_at) if body.punched_at is not None else datetime.now(UTC).replace(microsecond=0)
    if body.punched_at is not None and (
        not isinstance(body.punched_at, int)
        or isinstance(body.punched_at, bool)
        or body.punched_at < 100000000000
        or body.punched_at > 4102444800000
    ):
        raise HTTPException(422, "Invalid timestamp")

    doc = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": now}},
        sort=[("punch_in", DESCENDING)],
    )
    if not doc:
        raise HTTPException(404, "No matching punch-in found")
    if doc.get("punch_out") is not None:
        raise HTTPException(409, "Already punched out")
    if now <= doc["punch_in"] or now > doc["punch_in"] + timedelta(hours=24):
        raise HTTPException(422, "Invalid punch-out time")

    hours = compute_work_hours(doc["punch_in"], now)
    overtime = compute_overtime(
        now, emp["shift_start"], emp["shift_end"], doc["date"]
    )
    db.attendance_logs.update_one(
        {"_id": doc["_id"], "punch_out": None},
        {"$set": {
            "punch_out": now,
            "work_hours": hours,
            "overtime_minutes": overtime,
            "half_day": hours < 4.50,
        }},
    )
    updated = db.attendance_logs.find_one({"_id": doc["_id"]})
    return attendance_response(updated)


@app.get("/attendance")
def list_attendance(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    emp_code: Optional[str] = None,
    department: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    filt = {}
    if start_date:
        try:
            date.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(422, "Invalid start_date")
        filt.setdefault("date", {})["$gte"] = start_date
    if end_date:
        try:
            date.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(422, "Invalid end_date")
        filt.setdefault("date", {})["$lte"] = end_date
    if start_date and end_date and start_date > end_date:
        raise HTTPException(422, "Invalid date range")
    if emp_code:
        filt["emp_code"] = emp_code
    if status:
        filt["status"] = status
    if department:
        employees = db.employees.distinct(
            "emp_code", {"department": department}
        )
        filt["emp_code"] = {"$in": employees}

    try:
        total = db.attendance_logs.count_documents(filt)
        docs = list(
            db.attendance_logs.find(filt)
            .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
    except PyMongoError:
        raise HTTPException(503, "Database unavailable")

    return {"items": [attendance_response(d) for d in docs],
            "total": total, "page": page, "page_size": page_size}
@app.get("/debug/db-check")
def db_check():
    try:
        client.admin.command("ping")
        return {
            "database": "connected",
            "employees": db.employees.count_documents({}),
            "attendance_logs": db.attendance_logs.count_documents({})
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str):
    import calendar

    try:
        month_start = date.fromisoformat(month + "-01")
        last_day = calendar.monthrange(
            month_start.year, month_start.month
        )[1]
        month_end = date(
            month_start.year, month_start.month, last_day
        )
    except (ValueError, TypeError):
        raise HTTPException(422, "month must be YYYY-MM")

    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(404, "Employee not found")

    joined = date.fromisoformat(emp["joined_on"]) if isinstance(
        emp["joined_on"], str
    ) else emp["joined_on"]

    first_workday = max(month_start, joined)
    working_days = sum(
        1 for day in range(1, last_day + 1)
        if date(month_start.year, month_start.month, day) >= first_workday
        and date(month_start.year, month_start.month, day).weekday() < 5
    )

    records = list(db.attendance_logs.find({
        "emp_code": emp_code,
        "date": {
            "$gte": month_start.isoformat(),
            "$lte": month_end.isoformat()
        }
    }))

    present_statuses = {"PRESENT", "WFH", "ON_DUTY"}
    present_days = sum(
        0.5 if r.get("half_day", False) else 1
        for r in records
        if r.get("status") in present_statuses
        and date.fromisoformat(r["date"]).weekday() < 5
    )

    late_records = [
        r for r in records if r.get("late_minutes", 0) > 0
    ]

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": round(present_days, 2),
        "leave_days": sum(
            1 for r in records if r.get("status") == "LEAVE"
        ),
        "late_count": len(late_records),
        "total_late_minutes": sum(
            r.get("late_minutes", 0) for r in records
        ),
        "total_overtime_minutes": sum(
            r.get("overtime_minutes", 0) for r in records
        ),
        "attendance_pct": (
            round(present_days / working_days * 100, 2)
            if working_days else None
        )
    }

@app.get("/analytics/departments/summary")
def department_summary(month: str, department: Optional[str] = None):
    import calendar

    try:
        month_start = date.fromisoformat(month + "-01")
        last_day = calendar.monthrange(
            month_start.year, month_start.month
        )[1]
        month_end = date(
            month_start.year, month_start.month, last_day
        ).isoformat()
    except ValueError:
        raise HTTPException(422, "month must be YYYY-MM")

    match = {"joined_on": {"$lte": month_end}}
    if department:
        match["department"] = department

    try:
        pipeline = [
            {"$match": match},
            {"$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "logs"
            }},
            {"$project": {
                "department": 1,
                "emp_code": 1,
                "logs": {
                    "$filter": {
                        "input": "$logs",
                        "as": "log",
                        "cond": {
                            "$and": [
                                {"$gte": ["$$log.date", month + "-01"]},
                                {"$lte": ["$$log.date", month_end]}
                            ]
                        }
                    }
                }
            }},
            {"$unwind": {
                "path": "$logs",
                "preserveNullAndEmptyArrays": True
            }},
            {"$group": {
                "_id": "$department",
                "headcount": {"$addToSet": "$emp_code"},
                "present_days": {"$sum": {
                    "$cond": [
                        {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                        {"$cond": [
                            {"$eq": ["$logs.half_day", True]}, 0.5, 1
                        ]},
                        0
                    ]
                }},
                "late_count": {"$sum": {
                    "$cond": [
                        {"$gt": [{"$ifNull": ["$logs.late_minutes", 0]}, 0]},
                        1, 0
                    ]
                }},
                "total_late_minutes": {"$sum": {
                    "$ifNull": ["$logs.late_minutes", 0]
                }},
                "leave_count": {"$sum": {
                    "$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]
                }},
                "on_duty_count": {"$sum": {
                    "$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]
                }},
                "work_hours": {"$push": {
                    "$cond": [
                        {"$and": [
                            {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                            {"$ne": ["$logs.work_hours", None]}
                        ]},
                        "$logs.work_hours", None
                    ]
                }}
            }},
            {"$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": {"$size": "$headcount"},
                "present_days": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
                "avg_work_hours": {
                    "$let": {
                        "vars": {
                            "hours": {
                                "$filter": {
                                    "input": "$work_hours",
                                    "as": "h",
                                    "cond": {"$ne": ["$$h", None]}
                                }
                            }
                        },
                        "in": {"$cond": [
                            {"$gt": [{"$size": "$$hours"}, 0]},
                            {"$round": [{"$avg": "$$hours"}, 2]},
                            None
                        ]}
                    }
                }
            }},
            {"$sort": {"department": 1}}
        ]
        results = list(db.employees.aggregate(pipeline))
    except PyMongoError:
        raise HTTPException(503, "Database unavailable")

    return {"month": month, "items": results}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str,
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
):
    import calendar

    try:
        month_start = date.fromisoformat(month + "-01")
        last_day = calendar.monthrange(
            month_start.year, month_start.month
        )[1]
        month_end = date(
            month_start.year, month_start.month, last_day
        ).isoformat()
    except ValueError:
        raise HTTPException(422, "month must be YYYY-MM")

    pipeline = [
        {"$match": {
            "date": {"$gte": month + "-01", "$lte": month_end},
            "late_minutes": {"$gt": 0},
        }},
        {"$group": {
            "_id": "$emp_code",
            "late_count": {"$sum": 1},
            "total_late_minutes": {"$sum": "$late_minutes"},
        }},
        {"$lookup": {
            "from": "employees",
            "localField": "_id",
            "foreignField": "emp_code",
            "as": "employee",
        }},
        {"$unwind": "$employee"},
    ]

    if department:
        pipeline.append({"$match": {"employee.department": department}})

    pipeline.extend([
        {"$setWindowFields": {
            "sortBy": {"total_late_minutes": -1},
            "output": {"rank": {"$rank": {}}},
        }},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {
            "_id": 0,
            "emp_code": "$_id",
            "name": "$employee.name",
            "department": "$employee.department",
            "late_count": 1,
            "total_late_minutes": 1,
            "rank": 1,
        }},
    ])

    try:
        items = list(db.attendance_logs.aggregate(pipeline))
    except PyMongoError:
        raise HTTPException(503, "Database unavailable")

    return {"month": month, "items": items}

@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    start_date: str,
    end_date: str,
):
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(422, "Dates must be YYYY-MM-DD")

    if end < start:
        raise HTTPException(422, "Invalid date range")
    if (end - start).days > 91:
        raise HTTPException(422, "Maximum range is 92 days")

    if db.employees.count_documents({"department": department}) == 0:
        raise HTTPException(404, "Department not found")

    dates = [
        (start + timedelta(days=i)).isoformat()
        for i in range((end - start).days + 1)
    ]

    pipeline = [
        {"$match": {
            "date": {"$gte": start_date, "$lte": end_date}
        }},
        {"$lookup": {
            "from": "employees",
            "localField": "emp_code",
            "foreignField": "emp_code",
            "as": "employee",
        }},
        {"$unwind": "$employee"},
        {"$match": {"employee.department": department}},
        {"$group": {
            "_id": "$date",
            "present_count": {"$sum": {
                "$cond": [
                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                    {"$cond": [
                        {"$eq": ["$half_day", True]}, 0.5, 1
                    ]},
                    0
                ]
            }},
            "late_count": {"$sum": {
                "$cond": [
                    {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]},
                    1, 0
                ]
            }},
        }},
    ]

    try:
        grouped = {
            x["_id"]: x for x in db.attendance_logs.aggregate(pipeline)
        }
        employees = list(db.employees.find(
            {"department": department},
            {"_id": 0, "emp_code": 1, "joined_on": 1},
        ))
    except PyMongoError:
        raise HTTPException(503, "Database unavailable")

    rows = []
    for day in dates:
        current = date.fromisoformat(day)
        headcount = sum(
            1 for e in employees
            if str(e["joined_on"]) <= day
        )
        data = grouped.get(day, {})
        present = data.get("present_count", 0)
        working = current.weekday() < 5
        rate = (
            round(present / headcount, 4)
            if working and headcount else None
        )
        rows.append({
            "date": day,
            "is_working_day": working,
            "headcount": headcount,
            "present_count": present,
            "late_count": data.get("late_count", 0),
            "attendance_rate": rate,
        })

    for i, row in enumerate(rows):
        recent = [
            r["attendance_rate"]
            for r in rows[max(0, i - 6):i + 1]
            if r["attendance_rate"] is not None
        ]
        row["moving_avg_7d"] = (
            round(sum(recent) / len(recent), 4) if recent else None
        )

    return {"department": department, "items": rows}
# ---------------------------------------------------------------------------
# Attendance regularization
# ---------------------------------------------------------------------------

from pymongo import ReturnDocument


class RegularizationRequest(BaseModel):
    by: str
    reason: str
    changes: dict


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(
    emp_code: str,
    date: str,
    body: RegularizationRequest,
):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, "Date must be YYYY-MM-DD")

    allowed_changes = {"punch_in", "punch_out", "status"}
    if not body.changes or not set(body.changes).issubset(allowed_changes):
        raise HTTPException(422, "Invalid changes")

    if not body.by.strip() or not body.reason.strip():
        raise HTTPException(422, "by and reason are required")

    doc = db.attendance_logs.find_one({
        "emp_code": emp_code,
        "date": date,
    })
    if not doc:
        raise HTTPException(404, "Attendance record not found")

    updates = {}
    history_changes = {}

    for key, value in body.changes.items():
        old_value = doc.get(key)

        if key in ("punch_in", "punch_out"):
            if value is not None:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise HTTPException(422, f"{key} must be epoch milliseconds")
                value = from_epoch_ms(value)

        if key == "status" and value not in ("PRESENT", "WFH", "ON_DUTY"):
            raise HTTPException(422, "Invalid status")

        updates[key] = value
        history_changes[key] = {
            "from": old_value,
            "to": value,
        }

    history_entry = {
        "at": datetime.now(UTC).replace(microsecond=0),
        "by": body.by,
        "reason": body.reason,
        "changes": history_changes,
    }

    try:
        updated = db.attendance_logs.find_one_and_update(
            {"_id": doc["_id"]},
            {
                "$set": updates,
                "$push": {"history": history_entry},
            },
            return_document=ReturnDocument.AFTER,
        )
    except PyMongoError:
        raise HTTPException(503, "Database unavailable")

    return attendance_response(updated)


# ---------------------------------------------------------------------------
# Query execution statistics
# ---------------------------------------------------------------------------

@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    month: Optional[str] = None,
    emp_code: Optional[str] = None,
    department: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    allowed = {
        "attendance_list",
        "employee_monthly",
        "department_summary",
        "late_leaderboard",
        "department_trend",
    }

    if endpoint not in allowed:
        raise HTTPException(
            status_code=404,
            detail="Unknown endpoint",
        )

    if endpoint in ("employee_monthly", "department_summary",
                    "late_leaderboard") and not month:
        raise HTTPException(422, "month is required")

    if endpoint == "employee_monthly" and not emp_code:
        raise HTTPException(422, "emp_code is required")

    if endpoint == "department_trend" and (
        not department or not start_date or not end_date
    ):
        raise HTTPException(
            422,
            "department, start_date and end_date are required",
        )

    try:
        if endpoint == "attendance_list":
            filt = {}
            if start_date:
                filt["date"] = {"$gte": start_date}
            if end_date:
                filt.setdefault("date", {})["$lte"] = end_date
            if emp_code:
                filt["emp_code"] = emp_code

        elif endpoint == "employee_monthly":
            filt = {
                "emp_code": emp_code,
                "date": {
                    "$gte": month + "-01",
                    "$lte": month + "-31",
                },
            }

        elif endpoint == "department_summary":
            employees = db.employees.distinct(
                "emp_code",
                {"department": department} if department else {},
            )
            filt = {
                "emp_code": {"$in": employees},
                "date": {
                    "$gte": month + "-01",
                    "$lte": month + "-31",
                },
            }

        elif endpoint == "late_leaderboard":
            filt = {
                "date": {
                    "$gte": month + "-01",
                    "$lte": month + "-31",
                },
                "late_minutes": {"$gt": 0},
            }

        else:
            filt = {
                "date": {
                    "$gte": start_date,
                    "$lte": end_date,
                },
            }
            employees = db.employees.distinct(
                "emp_code",
                {"department": department},
            )
            filt["emp_code"] = {"$in": employees}

        result = db.command(
            "explain",
            {
                "find": "attendance_logs",
                "filter": filt,
            },
            verbosity="executionStats",
        )

        return {
            "endpoint": endpoint,
            "query": filt,
            "executionStats": result.get("executionStats", {}),
            "queryPlanner": result.get("queryPlanner", {}),
        }

    except PyMongoError:
        raise HTTPException(503, "Database unavailable")
