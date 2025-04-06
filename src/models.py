from sqlalchemy import Column, Integer, String, Text, ForeignKey, Date, JSON, func
from sqlalchemy.orm import relationship
from src.database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)

class Package(Base):
    __tablename__ = 'packages'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

class Task(Base):
    __tablename__ = 'tasks'
    id = Column(Integer, primary_key=True, index=True)
    department_id = Column(Integer, ForeignKey('departments.id'))
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

class TaskPackage(Base):
    __tablename__ = 'task_package'
    task_id = Column(Integer, ForeignKey('tasks.id'), primary_key=True)
    package_id = Column(Integer, ForeignKey('packages.id'), primary_key=True)

class Department(Base):
    __tablename__ = 'departments'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

class DepartmentTeam(Base):
    __tablename__ = 'departments_has_teams'
    department_id = Column(Integer, ForeignKey('departments.id'), primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'), primary_key=True)

class Team(Base):
    __tablename__ = 'teams'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)

class TeamUser(Base):
    __tablename__ = 'users_has_teams'
    team_id = Column(Integer, ForeignKey('teams.id'), primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)

class Project(Base):
    __tablename__ = 'projects'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    start = Column(Date, nullable=False)
    end = Column(Date, nullable=False)

class ProjectTeam(Base):
    __tablename__ = 'project_teams'
    project_id = Column(Integer, ForeignKey('projects.id'), primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'), primary_key=True)

class Category(Base):
    __tablename__ = 'task_category'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)

class TeamAllocation(Base):
    __tablename__ = 'team_allocations'
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete="CASCADE"), nullable=False)
    package_id = Column(Integer, ForeignKey('packages.id', ondelete="CASCADE"), nullable=False)
    start_date = Column(Date, default=func.current_date, nullable=False)
    end_date = Column(Date, nullable=False)
    allocated_teams = Column(JSON, nullable=True)

class Skill(Base):
    __tablename__ = 'skills'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)

class UserSkill(Base):
    __tablename__ = 'user_skills'
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    skill_id = Column(Integer, ForeignKey('skills.id'), primary_key=True)

class TrelloProjectTask(Base):
    __tablename__ = 'trello_project_tasks'
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('projects.id'), nullable=False)
    trello_board_data = Column(JSON, nullable=True)
    start_date = Column(Date, nullable=False)
    event_date = Column(Date, nullable=False)

