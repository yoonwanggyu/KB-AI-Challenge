

create table dept(
d_id	int			not null	primary key,
d_name	varchar(20)	not null
);

create table empl(
e_id	int			not null	primary key,
e_name	varchar(20),
e_mail	varchar(20),
d_id	int			not null,
constraint e_fk foreign key(d_id) references dept(d_id)
);	

create table univ(
dept_id		int			not null primary key,
dept_name	varchar(20)
);

create table student(
student_id		int			not null primary key,
student_name	varchar(20),
dept_id			int,
student_email	varchar(20),
constraint e_fk foreign key (dept_id) references univ(dept_id)
);


