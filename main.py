import secrets
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import stripe
import tiktoken as tiktoken
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, desc,DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy.sql import func
from typing import Optional, Tuple
from datetime import datetime
from typing import List,Dict
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from passlib.context import CryptContext
from fastapi.responses import RedirectResponse

load_dotenv()  # take environment variables from .env.
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

from CreateKnowledgeStore import create_chatbot, GetDocumentsFromURL
from DeleteKnowledgeStore import deleteVectorsusingKnowledgeBaseID
from ChatChain import Get_Conversation_chain

# Create SQLite database engine
SQLALCHEMY_DATABASE_URL = "sqlite:///./Database.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for SQLAlchemy models
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Define SQLAlchemy model for the entry
class Users(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    name= Column(String)
    email = Column(String, unique=True, index=True)
    password=Column(String)
    isVerified=Column(Boolean,default=False)
    verification_token = Column(String, unique=True, nullable=True)

class Selected_Plan(Base):
    __tablename__ = "Selected_Plan"
    Selected_Plan_ID = Column(Integer, primary_key=True, index=True)
    plan_id= Column(String)
    user_id= Column(String)
    purchased_at = Column(DateTime, default=datetime.utcnow)
    last_updated= Column(DateTime, default=datetime.utcnow)
    status=Column(Boolean, default=True)

class Consumption(Base):
    __tablename__ = "Consumption"
    consumption_id = Column(Integer, primary_key=True, index=True)
    plan_id=Column(Integer)
    user_id=Column(Integer)
    consumed_chatbots=Column(Integer,default=0)
    consumed_store_tokens=Column(Integer,default=0)
    consumed_chatbot_response_tokens=Column(Integer,default=0)
    last_updated= Column(DateTime, default=func.now())

class Plans(Base):
    __tablename__ = "Plans"
    plan_id = Column(Integer, primary_key=True, index=True)
    plan_Name = Column(String)
    total_chatbots_allowed=Column(Integer)
    total_knowldegStores_Allowed_Tokens=Column(Integer)
    Total_Responce_Tokens_allowed=Column(Integer)
    price=Column(Integer)
    added_at = Column(DateTime, default=func.now())

class ChatBotsConfigurations(Base):
    __tablename__ = "chatBots_configurations"
    chatbot_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    descriptive_name= Column(String)
    temperature = Column(String)
    llm = Column(String)
    urls=Column(String)

class ChatbotAppearnace(Base):
    __tablename__ = "ChatbotAppearnace"
    chatbot_appeance_id = Column(Integer, primary_key=True, index=True)
    chatbot_id = Column(Integer)
    ThemeColor=Column(String)
    InitialMessage=Column(String)
    DisplayName=Column(String)


class ChatLogs(Base):
    __tablename__ = "ChatLogs"
    Message_id = Column(Integer, primary_key=True, index=True)
    chatbot_id = Column(Integer)
    visitor_ID = Column(Integer)
    Human_Message = Column(String)
    AI_Responce = Column(String)
    context = Column(String)
    responded_at = Column(DateTime, default=datetime.utcnow)

class LeadsGenerated(Base):
    __tablename__ = "LeadsGenerated"
    generated_leads_id = Column(Integer, primary_key=True, index=True)
    chatbot_id = Column(Integer)
    name  = Column(String)
    email = Column(String)
    phone = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class PaymentsTransactions(Base):
    __tablename__ = "PaymentsTransactions"
    payment_id = Column(Integer, primary_key=True, index=True)
    user_id=Column(Integer)
    transaction_id=Column(String)
    plan_id=Column(Integer)
    paymentMethodId=Column(String)
    payment_amount=Column(Integer)
    transactionTYpe=Column(String, default="First Time")
    tarnsactiondate = Column(DateTime, default=datetime.utcnow)

# Create tables in the database
Base.metadata.create_all(bind=engine)

class user(BaseModel):
    name:str
    email:str
    password:str

class ChatBots(BaseModel):
    user_id :int
    descriptive_name:str
    temperature:str
    llm:str
    urls: str

class EditChatBots(BaseModel):
    descriptive_name:str
    temperature:str
    llm:str


class EditAppeanceChatBots(BaseModel):
    ThemeColor:str
    InitialMessage:str
    DisplayName:str

class AddLeadsPydanticModel(BaseModel):
    chatbot_id:int
    name: Optional[str]=""
    email: Optional[str]=""
    phone: Optional[str]=""

class PlansPydnaticModel(BaseModel):
    plan_Name:str
    total_chatbots_allowed:int
    total_knowldegStores_Allowed_Tokens:int
    Total_Responce_Tokens_allowed:int
    price:int

class ChatRequest(BaseModel):
    chatbotId: int
    question: str
    visitorID:int
    chat_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    answer: str
    reference_context: List[dict]

class RenewPlan(BaseModel):
    Selected_Plan_ID:int
    transaction_id:int
    paymentMethodId:int
    payment_amount:int

class SubscribePlan(BaseModel):
    plan_id:int
    user_id:int
    transaction_id:str
    paymentMethodId:str
    payment_amount:int

class CheckPlanExistance(BaseModel):
    plan_id: int
    user_id: int

class PaymentIntentResult(BaseModel):
    clientSecret: str

class PaymentIntent(BaseModel):
    amount:int

# Stripe API key setup
stripe.api_key = os.getenv('STRIPE_KEY')


# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
BASE_URL="https://saas-oa-api.aishteck.com/"
BASE_URL="http://127.0.0.1:8000/"

@app.post("/SubscribePlanByUserID/")
async def SubscribePlanByUserID(request: SubscribePlan):
    try:
        db = SessionLocal()
        planExists = db.query(Plans).filter(Plans.plan_id == request.plan_id).first()
        if planExists:
            existing_plan = db.query(Selected_Plan).filter(Selected_Plan.plan_id == request.plan_id,
                                                                  Selected_Plan.user_id == request.user_id).first()
            if not existing_plan:
                plan_entry = Selected_Plan(user_id=request.user_id, plan_id=request.plan_id)
                db.add(plan_entry)
                db.commit()
                transactionEntry=PaymentsTransactions(**request.dict())
                db.add(transactionEntry)
                db.commit()
                plan_consumpiton_entry = Consumption(user_id=request.user_id, plan_id=request.plan_id)
                db.add(plan_consumpiton_entry)
                db.commit()

                db.close()
                return {"status": "ok", "message": "Plan Added to the Your Plans Pool successfully.", "data": None}
            else:
                raise HTTPException(status_code=400, detail="Provided Plan Exists for the user You cannot Stack Same Plans Try another Plan.")
        else:
            raise HTTPException(status_code=400, detail="Provided Plan Do not Exists.")

    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}

@app.post("/create-payment-intent", response_model=PaymentIntentResult)
async def create_payment_intent(request:PaymentIntent):
    try:
        # Create a payment intent with the specified amount and currency
        payment_intent = stripe.PaymentIntent.create(
            amount=request.amount,  # amount in cents
            currency="usd"
        )
        return PaymentIntentResult(clientSecret=payment_intent.client_secret)
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/checkifPlanExists/")
async def checkifPlanExists(request: CheckPlanExistance):
    try:
        db = SessionLocal()
        planExists = db.query(Plans).filter(Plans.plan_id == request.plan_id).first()
        if planExists:
            existing_plan = db.query(Selected_Plan).filter(Selected_Plan.plan_id == request.plan_id,
                                                                  Selected_Plan.user_id == request.user_id).first()
            if not existing_plan:
                db.close()
                return {"status": "ok", "message": "You can add this Plan it doesnt exist in plan pool.", "data": {"PlanStack":True}}
            else:
                return {"status": "ok", "message": "Plan Already Existed cannot stack Multiple Plans Try another Plan.", "data": {"PlanStack":False}}
        else:
            raise HTTPException(status_code=400, detail="Provided Plan Do not Exists.")

    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}

@app.get("/GetAllPlans/")
async def GetAllPlans():
    try:
        db = SessionLocal()
        entries = db.query(Plans).all()
        if entries:
            db.close()
            return {"status": "ok", "message": "Our Plans Data returned Successfully.", "data": entries}
        else:
            raise HTTPException(status_code=404, detail="Our Plans not found.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.delete("/DeletePlanById/{plan_id}")
def DeletePlanById(plan_id: int):
    try:
        db = SessionLocal()
        # Check if the chatbot exists
        ourplans_object = db.query(Plans).filter(Plans.plan_id == plan_id).first()
        if not ourplans_object:
            raise HTTPException(status_code=404, detail="Chatbot with the given ID not found.")
        db.delete(ourplans_object)
        db.commit()
        return {"status": "ok", "message": "Plan Data deleted successfully.", "data": None}

    except Exception as e:
        db.rollback()  # Rollback changes if an error occurs
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        db.close()

def getAggreatedPlansConsumptionByUsersPlanList(db,user_id):
    entries = db.query(Selected_Plan).filter(Selected_Plan.user_id == user_id).all()
    if not entries:
        raise HTTPException(status_code=404, detail="No Plans Found under Given user ID.")
    accamulated_chatbots = 0
    accamulated_responce_tokens_credits = 0
    accamulated_knowledgeStore_Tokens = 0
    accamulated_price_paid = 0
    total_active_plans = 0
    for entry in entries:
        planInfo = db.query(Plans).filter(Plans.plan_id == entry.plan_id).first()
        if entry.status:
            total_active_plans += 1
            accamulated_chatbots += planInfo.total_chatbots_allowed
            accamulated_responce_tokens_credits += planInfo.Total_Responce_Tokens_allowed
            accamulated_knowledgeStore_Tokens += planInfo.total_knowldegStores_Allowed_Tokens
            accamulated_price_paid += planInfo.price

    aggregatedPlan={
                        "user_id": user_id,
                        "Total_Plans": len(entries),
                        "Total_active_Plans": total_active_plans,
                        "Total_inactive_Plans": len(entries) - total_active_plans,
                        "Aggregated_Chatbots":accamulated_chatbots,
                        "Aggregated_Monthly_Chatbot_Response_Tokens":accamulated_responce_tokens_credits,
                        "Aggregated_Knowledge_Store_Tokens":accamulated_knowledgeStore_Tokens ,
                        "Aggregated_Cost":accamulated_price_paid,

                    }
    aggreagate_consumption=db.query(Consumption).filter(Consumption.user_id == user_id).first()
    return  aggregatedPlan,aggreagate_consumption

def getAggreatedPlansInfoByUsersPlanList(db,user_id,entries):
    accamulated_chatbots = 0
    accamulated_responce_tokens_credits = 0
    accamulated_knowledgeStore_Tokens = 0
    accamulated_price_paid = 0
    total_active_plans=0
    for entry in entries:
        planInfo = db.query(Plans).filter(Plans.plan_id == entry.plan_id).first()
        if entry.status:
            total_active_plans+=1
            accamulated_chatbots += planInfo.total_chatbots_allowed
            accamulated_responce_tokens_credits += planInfo.Total_Responce_Tokens_allowed
            accamulated_knowledgeStore_Tokens += planInfo.total_knowldegStores_Allowed_Tokens
            accamulated_price_paid += planInfo.price
        entry.PlanInfo = planInfo
        TrasnactionOfPlan = db.query(PaymentsTransactions).filter(PaymentsTransactions.plan_id == entry.plan_id,
                                                                  PaymentsTransactions.user_id==user_id).all()
        if TrasnactionOfPlan:
            entry.TransactionHistory = TrasnactionOfPlan
        else:
            entry.TransactionHistory = "Basic Plan Attached (No History)"

    aggregatedPlan={
                        "user_id":user_id,
                        "Total_Plans": len(entries),
                        "Total_active_Plans": total_active_plans,
                        "Total_inactive_Plans": len(entries) - total_active_plans,
                        "Aggregated_Chatbots":accamulated_chatbots,
                        "Aggregated_Monthly_Chatbot_Response_Tokens":accamulated_responce_tokens_credits,
                        "Aggregated_Knowledge_Store_Tokens":accamulated_knowledgeStore_Tokens ,
                        "Aggregated_Cost":accamulated_price_paid,
                    }
    userinfo = db.query(Users).filter(Users.user_id == user_id).first()
    aggreagate_consumption=db.query(Consumption).filter(Consumption.user_id == user_id).first()
    return    {
        "User_Information": userinfo,
        "Individual_Plans_Information": entries,
        "Aggregated_Plans_Information":aggregatedPlan,
        "Aggregated_Consumption_Qouta":aggreagate_consumption
    }

def updatetheConsumptionAfterPlanInACtivation(db,purchasedPlanObject):
    planInfo = db.query(Plans).filter(Plans.plan_id == purchasedPlanObject.plan_id).first()
    consumption = db.query(Consumption).filter(Consumption.user_id == purchasedPlanObject.user_id).first()
    consumption.consumed_chatbots = max(0, consumption.consumed_chatbots - planInfo.total_chatbots_allowed)
    consumption.consumed_store_tokens = max(0,
                                            consumption.consumed_store_tokens - planInfo.total_knowldegStores_Allowed_Tokens)
    consumption.consumed_chatbot_response_tokens = max(0,
                                                       consumption.consumed_chatbot_response_tokens - planInfo.Total_Responce_Tokens_allowed)
    consumption.last_updated = datetime.utcnow()
    db.add(consumption)
    db.add(consumption)
    db.commit()
@app.get("/MakeUserPlanInactiveByPlanPurchaseId/{Selected_Plan_ID}")
async def MakeUserPlanInactiveByPlanPurchaseId(Selected_Plan_ID: int):
    try:
        db = SessionLocal()
        entry = db.query(Selected_Plan).filter(Selected_Plan.Selected_Plan_ID == Selected_Plan_ID).first()
        if entry:
            if entry.status:
                entry.status=False
                entry.last_updated=datetime.utcnow()
                db.add(entry)
                db.commit()
                updatetheConsumptionAfterPlanInACtivation(db,entry)
                db.close()
                return {"status": "ok", "message": "Given Plan set to be Inactive.", "data": None}
            else:
                raise HTTPException(status_code=404, detail="Give Plan is already Inactive.")
        else:
            raise HTTPException(status_code=404, detail="User Plan information not found under Given ID.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.post("/RenewPlanByPlanPurchaseId/")
async def RenewPlanByPlanPurchaseId(request: RenewPlan):
    try:
        db = SessionLocal()
        entry = db.query(Selected_Plan).filter(Selected_Plan.Selected_Plan_ID == request.Selected_Plan_ID).first()
        if entry:
            if not entry.status:
                entry.status=True
                entry.last_updated=datetime.utcnow()
                db.add(entry)
                db.commit()
                db.refresh(entry)
                transactionEntry = PaymentsTransactions(user_id=entry.user_id,plan_id=entry.plan_id,transactionTYpe="Renewel",transaction_id=request.transaction_id,paymentMethodId=request.paymentMethodId,payment_amount=request.payment_amount)
                db.add(transactionEntry)
                db.commit()
                db.close()
                return {"status": "ok", "message": "Given Plan set to be Active.", "data": None}
            else:
                raise HTTPException(status_code=404, detail="Give Plan is already Active.")
        else:
            raise HTTPException(status_code=404, detail="User Plan information not found under Given ID.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetAllPlansByUserId/{user_id}")
async def GetAllPlansByUserId(user_id: int):
    try:
        db = SessionLocal()
        entries = db.query(Selected_Plan).filter(Selected_Plan.user_id==user_id).all()
        if entries:
            response=getAggreatedPlansInfoByUsersPlanList(db, user_id, entries)
            db.close()
            return {"status": "ok", "message": "Plans Data returned Successfully.", "data": response}
        else:
            raise HTTPException(status_code=404, detail="No Plans Found under GIven user ID.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.post("/AddPlanToOurPlans/")
async def AddPlanToOurPlans(request: PlansPydnaticModel):
    try:
        db = SessionLocal()
        db_entry = Plans(**request.dict())
        db.add(db_entry)
        db.commit()
        db.close()
        return {"status": "ok", "message": "Plan Added to the Our Plans successfully.", "data": None}
    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}

@app.post("/register/")
def register(request: user):
    try:
        db = SessionLocal()
        existing_user = db.query(Users).filter(Users.email == request.email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered Please use any other Email Address or Login with Existing Email Address.")
        request.password = pwd_context.hash(request.password)
        db_entry = Users(**request.dict())
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        db_entry.verification_token=generate_verification_token()
        db.commit()
        send_verifiaction_code_on_email(db_entry.email,db_entry.name,db_entry.verification_token)
        db.commit()
        db.close()
        return {"status": "ok",
                    "message": "User registered successfully. Check your email for verification instructions.",
                    "data": db_entry}

    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}

@app.post("/login/")
def login(email: str, password: str):
    try:
        db = SessionLocal()
        user = db.query(Users).filter(Users.email == email).first()
        if user:
            if user.isVerified:
                if not user or not pwd_context.verify(password, user.password):
                    raise HTTPException(status_code=401, detail="Invalid Credentials.")
                user_dict = {"user_id": user.user_id, "name": user.name, "email": user.email}
                db.close()
                return {"status": "ok", "message": "Account has been Authenticated.", "data": user_dict}
            else:
                send_verifiaction_code_on_email(user.email, user.name, user.verification_token)
                return {"status": "ok", "message": "Your account is not Verified Please Check Your Email Address for Verification Link.", "data": None}
        else:
            return {"status": "error",
                    "message": "User not Found.",
                    "data": None}
    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}

def generate_verification_token():
    return secrets.token_urlsafe(16)

def send_verifiaction_code_on_email(receiver,good_name,verification_token):

    port = 587  # For starttls
    smtp_server = "cloudhost-4480981.us-midwest-1.nxcli.net"
    sender_email = "support@optimalaccess.com"
    receiver_email = receiver
    subject = "Account Verification Link (WebSync From Maktek)"
    password = "StakedMethodRoodTannin"

    body="""<!DOCTYPE html>
    <html>
      <head>
        <style>
          * {
            font-family: "Montserrat", sans-serif;
            color:white;
          }
          ul li
            {
                margin-bottom:5px;
            }
          body {
            font-family: Arial, sans-serif;
            background-color: gray;
            color: white;
            font-family: "Montserrat", sans-serif;
            padding:20px;
          }
          .container {
            max-width: 100%;
            color: white;
            margin: 0 auto;
            padding: 20px;
            border: 1px solid white;
            background-color: gray;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
          }

          .footer-like {
            margin-top: auto;
            padding: 6px;
            text-align: center;
          }
          .footer-like p {
            margin: 0;
            padding: 4px;
            color: #fafafa;
            font-family: "Raleway", sans-serif;
            letter-spacing: 1px;
          }
          .footer-like p a {
            text-decoration: none;
            font-weight: 600;
          }

          .logo {
            width: 100px;
            border:1px solid white;
          }
          .verify-button
          {
          text-decoration:none;
          background-color:white;
          border-radius:5px;
          padding:10px;
          border: none;
          text-decoration:none;
          }
        </style>
      </head>
      <body>
        <div class="container">
    <img src="https://i.ibb.co/2k2YhLC/image-2-removebg-preview.png" alt="Maktek-Logo" border="0" class="logo" />
    """
    body += f'<p>Dear {good_name},</p>' \
            f'<h1><strong>Welcome to WebSync!</strong></h1>' \
            f'<p>Your Account Verification Link is placed below. Click on the link to get verified:</p>' \
            f'<h4><b><a href="{BASE_URL}verify?token={verification_token}" class="verify-button" style="color:gray;">Click here to Verify Your Account</a></b></h4>'
    body+="""
          <p><b>Sincerely,</b><br />The WebSync Team</p>
          <div class="footer-like">
            <p>
              Powered by Maktek
            </p>
          </div>
        </div>
      </body>
    </html>"""

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to email
    message.attach(MIMEText(body, "html"))

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.starttls(context=context)  # Secure the connection
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()

@app.get("/resendVerificationToken/")
def resendVerificationToken(email: str):
    try:
        db = SessionLocal()
        user = db.query(Users).filter(Users.email == email).first()
        if user:
            if user.isVerified:
                db.close()
                return {"status": "ok", "message": "Account is already Verified.", "data": None}
            else:
                send_verifiaction_code_on_email(user.email, user.name, user.verification_token)
                return {"status": "ok", "message": "Verification Link has been Resent to your Email Address.", "data": None}
        else:
            return {"status": "error",
                    "message": "Provided Email Address does not points to any Registered account.",
                    "data": None}
    except Exception as e:
            db.close()
            return {"status": "error","message": str(e), "data": None}

def generate_verification_code():
    # Generate a random 6-digit hexadecimal code
    verification_code = secrets.token_hex(3).upper()
    return verification_code

@app.get("/verify/")
def verify_account(token: str):
    try:
        db = SessionLocal()
        user = db.query(Users).filter(Users.verification_token == token).first()
        if user:
            if user.isVerified:
                db.close()
                return RedirectResponse(url="https://kchat.website/auth/signin/?message=Account+is+already+Verified.")  # Redirect to the given page with a message
                #return {"status": "ok", "message": "Account is already Verified.", "data": None}
            else:
                user.isVerified=True
                db.commit()
                db.close()
                return RedirectResponse(url="https://kchat.website/auth/signin/?message=Your+account+has+been+verified+login+to+access+Dashboard.")  # Redirect to the given page with a message
        else:
            return {"status": "error",
                    "message": "User not Found.",
                    "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.post("/SendVerificationEmail/")
async def SendVerificationEmail (receiver:str):
    send_verifiaction_code_on_email(receiver,"Ali Haider","TestToken")

def upadteChatlogs(db,chatbot_id,visitor_ID,Human_Message,AI_Responce,context):
    ChatLogs_Object = ChatLogs(chatbot_id=chatbot_id, visitor_ID=visitor_ID,
                               Human_Message=Human_Message, AI_Responce=AI_Responce,context=context)
    db.add(ChatLogs_Object)
    db.commit()
    return

@app.delete("/DeleteChatLogByChatID/{chatbot_id}")
async def DeleteChatLogByChatID(chatbot_id: int):
    db = SessionLocal()
    try:
        ChatLogs_info = db.query(ChatLogs).filter(ChatLogs.chatbot_id == chatbot_id)
        if ChatLogs_info.count() > 0:
            ChatLogs_info.delete()
            db.commit()
            db.close()
            return {"status": "ok", "message": "Chatbot's Chat Logs deleted successfully.", "data": None}
        else:
            raise HTTPException(status_code=404, detail="Chatbot ChatLogs not Found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
@app.get("/GetChatLogsByChatBotID/{chatbot_id}")
async def GetChatLogsByChatBotID(chatbot_id: int):
    try:
        db = SessionLocal()
        chatlogs = db.query(ChatLogs).filter(ChatLogs.chatbot_id == chatbot_id).all()
        if not chatlogs:
            raise HTTPException(status_code=404, detail="Chatbot do not Contains any Chat Logs.")
        db.close()
        return {"status": "ok", "message": "Chatbots Chat Logs returned Successfully.", "data": chatlogs}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}
@app.get("/GetChatLogsByUserID/{user_id}")
async def GetChatLogsByUserID(user_id: int):
    try:
        db = SessionLocal()
        chatbots = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
        ids=[]
        for bot in chatbots:
            ids.append(bot.chatbot_id)

        chatlogs = db.query(ChatLogs).filter(ChatLogs.chatbot_id.in_(ids)).all()
        if not chatlogs:
            raise HTTPException(status_code=404, detail="User Chatbot do not Contains any Chat Logs.")
        db.close()
        return {"status": "ok", "message": "All Chatbots Chat Logs under given USer ID returned Successfully.", "data": chatlogs}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

def VerifyChatbotCreationQouta(db,user_id):
    Aggregated_Plans_Information,Aggregated_Consumption=getAggreatedPlansConsumptionByUsersPlanList(db,user_id)
    reamining_chatbots=Aggregated_Plans_Information['Aggregated_Chatbots']-Aggregated_Consumption.consumed_chatbots

    if reamining_chatbots<=0:
        return f"Chatbots Creation Quota Exceeds (Quota Left for Creation {max(reamining_chatbots,0)} )",False
    return "", True

def VerifyKnowldegeBaseTokensQouta(db,user_id,current_tokens_demanded):
    Aggregated_Plans_Information,Aggregated_Consumption=getAggreatedPlansConsumptionByUsersPlanList(db,user_id)
    reamining_knowledge_base_tokens = Aggregated_Plans_Information['Aggregated_Knowledge_Store_Tokens'] - Aggregated_Consumption.consumed_store_tokens
    responceText=""
    verified=True
    if current_tokens_demanded>reamining_knowledge_base_tokens:
        responceText+=f"Requested Knowledge Base Tokens Quota Exceeds Try Source with less Number of Tokens.\n"
        verified=False
    if reamining_knowledge_base_tokens<=0:
        responceText +=f"Knowledge Base Tokens Quota Exceeds (Remaining Quota {max(reamining_knowledge_base_tokens,0)} Tokens) update the Plan and Try Again.\n"
        verified=False
    return responceText, verified

def updateChatbotsCreationCount(db,user_id):
    comsuptionObj = db.query(Consumption).filter(Consumption.user_id == user_id).first()
    comsuptionObj.consumed_chatbots=comsuptionObj.consumed_chatbots+1
    db.add(comsuptionObj)
    db.commit()

def num_tokens_from_string(string: str, encoding_name="text-embedding-ada-002") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def VerifyChatbotResponceCreditQouta(db,user_id,currentTokens):
    Aggregated_Plans_Information,Aggregated_Consumption=getAggreatedPlansConsumptionByUsersPlanList(db,user_id)
    reamining_chatbots_responces=Aggregated_Plans_Information['Aggregated_Monthly_Chatbot_Response_Tokens']-Aggregated_Consumption.consumed_chatbot_response_tokens-currentTokens
    if reamining_chatbots_responces<=0:
        return f"Chatbots Response Credits Quota Exceeds (Response Quota Left {max(reamining_chatbots_responces,0)} ) Upgrade Plan for More Credits",False
    return "", True

def updateChatbotsResponseCreditCount(db,user_id,curremt_demanded_token):
    comsuptionObj = db.query(Consumption).filter(Consumption.user_id == user_id).first()
    comsuptionObj.consumed_chatbot_response_tokens=comsuptionObj.consumed_chatbot_response_tokens+curremt_demanded_token
    comsuptionObj.last_updated=datetime.utcnow()
    db.add(comsuptionObj)
    db.commit()

###################################
@app.post("/Chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        db = SessionLocal()
        db_entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == request.chatbotId).first()
        if db_entry:
            try:
                answer,sources=Get_Conversation_chain(db_entry.chatbot_id,db_entry.temperature,db_entry.llm,request.question,request.chat_history)
                currentTokens=num_tokens_from_string(str(request.question) + str(answer) + str(sources))
                verification = VerifyChatbotResponceCreditQouta(db, db_entry.user_id, currentTokens)
                if not verification[1]:
                    raise HTTPException(status_code=404, detail=f'{verification[0]}')

                upadteChatlogs(db,db_entry.chatbot_id,request.visitorID,request.question,answer,str(sources))
                updateChatbotsResponseCreditCount(db, db_entry.user_id,currentTokens)
                db.close()
                return ChatResponse(answer=answer, reference_context=sources)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            db.close()
            return ChatResponse(answer="Chatbot Configuration not Found under ID: " + str(request.chatbotId) , reference_context=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def updateKnowledgeBaseCreationandTokensCount(db,user_id,tokens_count):
    comsuptionObj = db.query(Consumption).filter(Consumption.user_id == user_id).first()
    comsuptionObj.consumed_store_tokens=comsuptionObj.consumed_store_tokens+tokens_count
    comsuptionObj.last_updated=datetime.utcnow()
    db.add(comsuptionObj)
    db.commit()

@app.post("/CreateChatbots/")
def createChatbot(entry: ChatBots):
    try:
        db = SessionLocal()
        verification = VerifyChatbotCreationQouta(db, entry.user_id)
        if not verification[1]:
            raise HTTPException(status_code=404, detail=f'{verification[0]}')

        db_entry = ChatBotsConfigurations(**entry.dict())
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        appaeranceEntry = ChatbotAppearnace(chatbot_id=db_entry.chatbot_id, ThemeColor="gray",
                                            InitialMessage=f"I am {db_entry.descriptive_name}, how can I help you?",
                                            DisplayName=db_entry.descriptive_name)
        db.add(appaeranceEntry)
        db.commit()
        db.refresh(appaeranceEntry)
        docs, tokenscount = GetDocumentsFromURL(entry.user_id, str(db_entry.chatbot_id), entry.urls)
        verification = VerifyKnowldegeBaseTokensQouta(db, entry.user_id,tokenscount)
        if not verification[1]:
            raise HTTPException(status_code=404, detail=f'{verification[0]}')

        if docs != False and tokenscount != -1:
            if create_chatbot(docs):
                updateKnowledgeBaseCreationandTokensCount(db, entry.user_id, tokenscount)
                updateChatbotsCreationCount(db, entry.user_id)
                db.close()
                return {"status": "ok", "message": "Chatbot Created Successfully.",
                        "data": {"Token_Tokens_consumed", tokenscount}}
    except Exception as e:
        last_entry = db.query(ChatBotsConfigurations).order_by(desc(ChatBotsConfigurations.chatbot_id)).first()
        if last_entry:
            db.delete(last_entry)
            db.commit()
        last_entry = db.query(ChatbotAppearnace).order_by(desc(ChatbotAppearnace.chatbot_id)).first()
        if last_entry:
            db.delete(last_entry)
            db.commit()
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.get("/GetChatBotApprancebyChatbotID/{chatbot_id}")
async def GetChatBotApprancebyChatbotID(chatbot_id: int):
    try:
        db = SessionLocal()
        chatbot = db.query(ChatbotAppearnace).filter(ChatbotAppearnace.chatbot_id == chatbot_id).first()
        if not chatbot:
            raise HTTPException(status_code=404, detail="Bot Appearance not found")
        db.close()
        return {"status": "ok", "message": "Appearance Returned Successfully.", "data": chatbot}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.put("/EditChatbotAppearance/{chatbot_id}")
def EditChatbotAppearance(chatbot_id: int, edited_Appearance_info: EditAppeanceChatBots):
    try:
        db = SessionLocal()
        db_entry = db.query(ChatbotAppearnace).filter(ChatbotAppearnace.chatbot_id == chatbot_id).first()
        if db_entry is None:
            raise HTTPException(status_code=404, detail="Chatbot Apperance Configuration not Found")

        for key, value in edited_Appearance_info.dict().items():
            setattr(db_entry, key, value)
        db.commit()
        db.refresh(db_entry)
        db.close()
        return {"status": "ok", "message": "Chatbot information updated successfully.", "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.put("/EditChatbot/{chatbot_id}")
def edit_chatbot(chatbot_id: int, edited_info: EditChatBots):
    try:
        db = SessionLocal()
        db_entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == chatbot_id).first()
        if db_entry is None:
            raise HTTPException(status_code=404, detail="Chatbot Configuration not Found")

        for key, value in edited_info.dict().items():
            setattr(db_entry, key, value)
        db.commit()
        db.refresh(db_entry)
        db.close()
        return {"status": "ok", "message": "Chatbot information updated successfully.", "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}


@app.get("/GetChatbotsbyUserID/{user_id}")
def get_chatbots_by_user_ID(user_id: int):
    try:
        db = SessionLocal()
        entries = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
        if entries:
            for entry in entries:
                entry.urls=entry.urls.split(',')
            db.close()
            return {"status": "ok", "message": "ChatBots Configurations returned Successfully.", "data": entries}
        else:
            raise HTTPException(status_code=404, detail="Chatbots Configuration under User ID ("+str(user_id)+") not found.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.get("/GetChatbotEmbedableScript/{chatbot_id}")
def get_chatbots_Embdeding_Script(chatbot_id: int):
    try:
        db = SessionLocal()
        entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == chatbot_id).first()
        if entry:
            script= f"""<script src='https://karan-api.000webhostapp.com/Chatbot.js'></script><script>setupChatbot({entry.chatbot_id});</script>"""
            return {"status": "ok", "message": "Chatbot Embed able SCript returned Successfully.", "data": {"script":script}}
        db.close()
        raise HTTPException(status_code=404, detail="Chatbot with ID: " + str(chatbot_id) + " not found")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}


@app.delete("/deleteChatbot/{chatbot_id}")
def delete_Chatbot(chatbot_id: int):
    try:
        db = SessionLocal()
        entry = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.chatbot_id == chatbot_id).first()
        if entry:
            deleteVectorsusingKnowledgeBaseID(chatbot_id)
            db.delete(entry)
            db.commit()
            chatbot = db.query(ChatbotAppearnace).filter(ChatbotAppearnace.chatbot_id == chatbot_id).first()
            db.delete(chatbot)
            db.commit()
            db.close()
            return {"status": "ok", "message": "Chatbot Deleted Successfully with ID:: " + str(
                chatbot_id), "data": None}
        else:
            raise HTTPException(status_code=404,
                                    detail="Chatbot with ID: " + str(chatbot_id) + " not found in Databse.")
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.post("/AddLeadsDataToChatBot/")
async def AddLeadsDataToChatBot(entry: AddLeadsPydanticModel):
    db=None
    try:
        db = SessionLocal()
        db_lead = LeadsGenerated(**entry.dict())
        db.add(db_lead)
        db.commit()
        db.refresh(db_lead)
        db.close()
        return {"status": "ok", "message": "Lead Successfully Added to the Chatbot.", "data": None}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.get("/GetLeadsGeneratedByChatBot/{chatbot_id}")
async def GetLeadsGeneratedByChatBot(chatbot_id: int):
    try:
        db = SessionLocal()
        leads_info = db.query(LeadsGenerated).filter(LeadsGenerated.chatbot_id == chatbot_id).all()
        if not leads_info:
            raise HTTPException(status_code=404, detail="Chatbot do not Contains any Leads.")
        db.close()
        return {"status": "ok", "message": "Chatbots Leads returned Successfully.", "data": leads_info}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.get("/GetLeadsGeneratedByUserID/{user_id}")
async def GetLeadsGeneratedByUserID(user_id: int):
    try:
        db = SessionLocal()
        chatbots = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
        ids = []
        for bot in chatbots:
            ids.append(bot.chatbot_id)
        leads_info = db.query(LeadsGenerated).filter(LeadsGenerated.chatbot_id.in_(ids)).all()
        if not leads_info:
            raise HTTPException(status_code=404, detail="User Chatbots do not Contains any Leads.")
        db.close()
        return {"status": "ok", "message": "Chatbots Leads under User ID returned Successfully.", "data": leads_info}
    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

def num_tokens_from_string(string: str, encoding_name="text-embedding-ada-002") -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@app.post("/GetLeadsGeneratedWithTime")
async def GetLeadsGeneratedWithTime(user_id: int,timeframe: str):
    try:
        db = SessionLocal()
        chatbots = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
        if not chatbots:
            raise HTTPException(status_code=404, detail="Chatbots not found under user ID")
        chatbotIds = []
        for chatbot in chatbots:
            chatbotIds.append(chatbot.chatbot_id)

        LeadswithtimeForChart = db.query(LeadsGenerated.generated_leads_id, LeadsGenerated.created_at).filter(
            LeadsGenerated.chatbot_id.in_(chatbotIds)).all()

        LeadswithtimeForChart_as_strings = [
            (generated_leads_id, created_at.strftime("%Y-%m-%d %H:%M:%S"))  # Format as desired
            for generated_leads_id, created_at in LeadswithtimeForChart]

        processed_data_for_leads_with_time_period = process_leads(LeadswithtimeForChart_as_strings)
        result_dict = {}
        for count, date_time in LeadswithtimeForChart_as_strings:
            result_dict[date_time] = result_dict.get(date_time, 0) + count

        if timeframe=="all_data":
            response_data = {
                "leads_with_time": result_dict
            }
        else:
            response_data = {
                "leads_with_time": (processed_data_for_leads_with_time_period[timeframe])
            }

        db.close()
        return {"status": "ok", "message": "Leads returned Successfully.", "data": response_data}

    except Exception as e:
        db.close()
        return {"status": "error", "message": str(e), "data": None}

@app.get("/GetChatBotsDashBoardByUserID/{user_id}")
async def GetChatBotsDashBoardByUserID(user_id: int):
        try:
            db = SessionLocal()
            chatbots = db.query(ChatBotsConfigurations).filter(ChatBotsConfigurations.user_id == user_id).all()
            if not chatbots:
                raise HTTPException(status_code=404, detail="Chatbots not found under user ID")
            total_chatbots = len(chatbots)
            total_output_tokens = 0
            total_leadsGenerated = 0
            total_input_tokens = 0
            sumofAllMEssages = 0
            chatbotIds = []
            for chatbot in chatbots:
                chatbotIds.append(chatbot.chatbot_id)
                total_leadsGenerated += db.query(func.count(LeadsGenerated.generated_leads_id).filter(
                    LeadsGenerated.chatbot_id == chatbot.chatbot_id)).scalar()
                chatlogs = db.query(ChatLogs).filter(ChatLogs.chatbot_id == chatbot.chatbot_id).all()
                sumofAllMEssages += len(chatlogs)
                for chatlog in chatlogs:
                    total_output_tokens += num_tokens_from_string(chatlog.AI_Responce)
                    total_input_tokens += num_tokens_from_string(chatlog.Human_Message)
                    total_input_tokens += num_tokens_from_string(chatlog.context)

            totalqureisresponded = db.query(func.count(ChatLogs.Message_id.distinct())).filter(
                ChatLogs.chatbot_id.in_(chatbotIds)).scalar()

            LeadswithtimeForChart = db.query(LeadsGenerated.generated_leads_id, LeadsGenerated.created_at).filter(
                LeadsGenerated.chatbot_id.in_(chatbotIds)).all()
            QuerieswithtimeForChart = db.query(ChatLogs.Message_id, ChatLogs.responded_at).filter(
                ChatLogs.chatbot_id.in_(chatbotIds)).all()

            LeadswithtimeForChart_as_strings = [
                (generated_leads_id, created_at.strftime("%Y-%m-%d %H:%M:%S"))  # Format as desired
                for generated_leads_id, created_at in LeadswithtimeForChart
            ]
            QuerieswithtimeForChart_as_strings = [
                (Message_id, responded_at.strftime("%Y-%m-%d %H:%M:%S"))  # Format as desired
                for Message_id, responded_at in QuerieswithtimeForChart
            ]
            processed_data_for_leads_with_time_period = process_leads(LeadswithtimeForChart_as_strings)
            print(processed_data_for_leads_with_time_period)
            response_data = {
                "Total_Chatbots": total_chatbots,
                "Queries_and_Responces_with_Time":QuerieswithtimeForChart_as_strings,
                "Total_Leads_Generated": total_leadsGenerated,
                "Leads_Generated_Raw_Time":LeadswithtimeForChart_as_strings,
                "Leads_Generated_with_Time_Period": processed_data_for_leads_with_time_period,
                "Total_Queries_Responded": totalqureisresponded,
                "Total_AI_Generated_Tokens": total_output_tokens,
                "Total_Input_Tokens_(Context_and_Question)": total_input_tokens,
                "Average_Number_of_Messages_Per_Chatbot": sumofAllMEssages / total_chatbots,
            }
            db.close()
            return {"status": "ok", "message": "Chatbots Dashboard returned Successfully.", "data": response_data}
        except Exception as e:
            db.close()
            return {"status": "error", "message": str(e), "data": None}
def process_leads(data: List[List]) -> Dict:
    year_data = {}
    month_data = {}
    day_data = {}

    for lead in data:
        lead_id = lead[0]
        timestamp_str = lead[1]
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day

            # Year data
            year_key = str(year)
            if year_key not in year_data:
                year_data[year_key] = 1
            else:
                year_data[year_key] += 1

            # Month data
            month_key = f"{year}-{month:02}"
            if month_key not in month_data:
                month_data[month_key] = 1
            else:
                month_data[month_key] += 1

            # Day data
            day_key = f"{year}-{month:02}-{day:02}"
            if day_key not in day_data:
                day_data[day_key] = 1
            else:
                day_data[day_key] += 1
        except ValueError:
            # Handle invalid timestamp format
            pass
    processed_data = {
        "year_data": year_data,
        "month_data": month_data,
        "day_data": day_data
    }
    return processed_data