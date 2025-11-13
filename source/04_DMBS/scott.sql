 -- [II] SELECT문 - 조회
--  SELECT * FROM TAB; -- 현계정(SCOTT)


-- 1. SELECT 문장 작성법
SELECT * FROM TAB; -- 현 계정이 가지고 있는 테이블 정보(실행:CTRL+ENTER)

SELECT * FROM DEPT; --DEPT 테이블의 모든 열, 모든 행

SELECT * FROM SALGRADE; -- SALGRADE 테이블의 모든 열, 모든 행

SELECT * FROM EMP;      -- EMP 테이블의 모든 열, 모든 행


-- 2. 특정 열만 출력
DESC EMP;
    -- EMP테이블의 구조
SELECT EMPNO, ENAME, SAL, JOB, FROM EMP; -- EMP테이블의 SELECT절에 지정된 열만 출력

SELECT EMPNO AS "사 번", ENAME AS 이름, SAL AS "급여", JOB AS "직책"
    FROM EMP; -- 열이름에 별칭을 두는 경우 EX. 열이름 "별칭", 열이름

SELECT EMPNO 사번, ENAME 이름, SAL 급여, JOB
    FROM EMP; -- 별칭에 SPAcE가 없는 경우는 따옴표 생략가능



-- 3. 특정 행 출력 : WHERE(조건절) -- 비교연산자 : 같다(=), 다르다(!=, ^=, <>), .,..
SELECt EMPNO 사번, ENAME 이름, SAL 급여 FROM EMP WHERE SAL=3000;
SELECT EMPNO NO, ENAME NAME, SAL FROM EMP WHERE SAL!=3000;
SELECT EMPNO NO, ENAME NAME, SAL FROM EMP WHERE SAL^=3000;
SELECT EMPNO NO, ENAME NAME, SAL FROM EMP WHERE SAL<>3000;
    -- 비교연산자는 숫자, 문자, 날짜형 모두 가능
    -- ex1. 사원이름이 'A','B','C'로 시작하는 사원의 모든 열(필드)
    -- A< AA < AAA < AAA < ... < B < BA < ... < C
    SELECT * FROM EMP WHERE ENAME < 'D';
    -- ex2. 81년도 이전에 입사한 사원의 모든 열(필드)
    SELECT * FROM EMP WHERE HIREDATE < '81/01/01';
    -- ex3 부서번호(DEPTNO)가 10번인 사원의 모든 필드
    select * from emp where deptno=10;
    -- SQL문은 대소문자 구별없음. 데이터는 대소문자 구별
    -- ex4. 이름(ENAME)이 SCOTT인 직원의 모든 데이터(필드)
    SELECT * FROM emp WHERE ename = 'SCOTT';

-- 4. WHERE절(조건절)에 논리연산자 : AND OR  NOT
    -- ex1. 급여(SAL)가 2000이상 3000이하인 직원의 모든 필드
    SELECT * FROM EMP WHERE 2000<=SAL AND SAL<3000;
    -- ex2. 82년도 입사한 사원의 모든 필드
    SELECT * FROM EMP WHERE HIREDATE>+'82/01/01' AND HIREDATE<='82/12/31';
    
    -- 날짜 표기법 셋팅 (현재:RR/MM/DD)
    ALTER SESSION SET NLS_DATE_FORMAT = 'RR/MM/DD';
    SELECT * FROM EMP
        WHERE TO_CHAR(HIREDATE, 'RR/MM/DD') >= '82/01/01'
            AND TO_CHAR(HIREDATE, 'RR/MM/DD' )<= '82/12/31';
            
    -- EX3. 부서번호가 10이 아닌 직원의 모든 필드
    SELECT * FROM EMP WHERE DEPTNO != 10;
    SELECT * FROM EMP WHERE NOT DEPTNO = 10;
    
-- 5. 산술연산자 (SELECT절, WHERE절, ORDER BY 절)
SELECT EMPNO, ENAME, SAL "예전 월급", SAL*1.1 "현재월급" FROM EMP;
    -- ex1. 연봉이 10,000이상인 직원의 ENAME(이름), SAL(월급), 연봉(SAL*12) - 연봉순
    SELECT ENAME, SAL, SAL*12 ANNUALSAL
        FROM EMP
        WHERE SAL*12 > 10000
        ORDER BY ANNUALSAL; ORDER
    -- 산술연산의 결과는 NULL을 포함하면 결과도 NULL
    -- NVL(NULL일 수도 있는 필드명, 대체값)을 이용 : 필드명과 대체값은 타입이 일치
    -- ex2. 연봉이 20000이상인 직원의 이름, 월급, 상여, 연봉(SAL*12+COMM)
    SELECT ENAME, SAL, COMM, SAL*12+NVL(COMM,0) 연봉 FROM EMP;
    -- ex3. 모든 사원의 ENAME, MGR(상사사번)을 출력
    SELECT ENAME, NVL(TO_CHAR(MGR), 'CEO') MGR FROM EMP;
    DESC EMP;


-- 6. 연결연산자 (||) :필드 내용이나문자를 연결
SELECT ENAME || '은(는)' ||JOB FROM EMP;

-- 7. 중복제거(DISTINCT)
SELECT DISTINCT JOB FROM EMP;
SELECT DISTINCT DEPTNO FROM EMP;

<총 연습문제>
--1.	EMP 테이블에서 sal이 3000이상인 사원의 empno, ename, job, sal을 출력

DESC EMP;
 
--2.	EMP 테이블에서 empno가 7788인 사원의 ename과 deptno를 출력

SELECT ename,deptno FROM EMP WHERE empno = 7788;

--3.	연봉(SAL*12+COMM)이 24000이상인 사번, 이름, 급여 출력 (급여순정렬)

SELECT EMPNO 사번, ENAME 이름, SAL 급여 FROM EMP WHERE (SAL*12+COMM) >= 24000;

--4.	입사일이 1981년 2월 20과 1981년 5월 1일 사이에 입사한 사원의 사원명, 직책, 입사일을 출력 (단 hiredate 순으로 출력)

--5.	deptno가 10,20인 사원의 모든 정보를 출력 (단 ename순으로 정렬)

SELECT * FROM EMP WHERE deptno = 10 or deptno = 20;

--6.	sal이 1500이상이고 deptno가 10,30인 사원의 ename과 sal를 출력
-- (단 출력되는 결과의 타이틀을 employee과 Monthly Salary로 출력)

SELECT ename, sal FROM EMP WHERE sal >= 1500 AND ;

--7.	hiredate가 1982년인 사원의 모든 정보를 출력

SELECT * FROM EMP WHERE HIREDATE<='82/12/31' AND HIREDATE >= '82/01/01';

--8.	이름의 첫자가 C부터  P로 시작하는 사람의 이름, 급여 이름순 정렬

SELECT ENAME, SAL  FROM EMP WHERE ENAME >= 'C' AND ENAME <='P';

--9.	comm이 sal보다 10%가 많은 모든 사원에 대하여 이름, 급여, 상여금을 
--출력하는 SELECT 문을 작성

SELECT EMPNO, ENAME, SAL FROM EMP WHERE SAL>=1500 AND SAL<=3000;

--10.	job이 CLERK이거나 ANALYST이고 sal이 1000,3000,5000이 아닌 모든 사원의 정보를 출력

SELECT * FROM EMP WHERE (JOB = 'CLERK' OR JOB = 'ANALYST') AND sal = 1000  AND sal = 3000 AND sal = 5000 ;

--11.	ename에 L이 두 자가 있고 deptno가 30이거나 또는 mgr이 7782인 사원의 
--모든 정보를 출력하는 SELECT 문을 작성하여라.



--12.	입사일이 81년도인 직원의 사번, 사원명, 입사일, 업무, 급여를 출력




--1. emp 테이블의 구조 출력

DESC EMP;

--2. emp 테이블의 모든 내용을 출력 

SELECT * FROM EMP;

--3. 현 scott 계정에서 사용가능한 테이블 출력

SELECT * FROM TAB;

--4. emp 테이블에서 사번, 이름, 급여, 업무, 입사일 출력

SELECT EMPNO, ENAME, SAL, JOB, HIREDATE FROM EMP;

--5. emp 테이블에서 급여가 2000미만인 사람의 사번, 이름, 급여 출력

SELECT EMPNO, ENAME, SAL FROM EMP WHERE SAL < 2000;

--6. 입사일이 81/02이후에 입사한 사람의 사번, 이름, 업무, 입사일 출력

SELECT EMPNO, ENAME, JOB, HIREDATE FROM EMP
    WHERE TO_CHAR(HIREDATE, 'RR/MM') >= '81/02';

--7. 업무가 SALESMAN인 사람들 모든 자료 출력

SELECT * FROM EMP WHERE JOB = 'SALESMAN';

--8. 업무가 CLERK이 아닌 사람들 모든 자료 출력

SELECT * FROM EMP WHERE JOB != 'CLERK';

--9. 급여가 1500이상이고 3000이하인 사람의 사번, 이름, 급여 출력

SELECT * FROM EMP WHERE SAL >= 1500 AND SAL <= 3000;

--10. 부서코드가 10번이거나 30인 사람의 사번, 이름, 업무, 부서코드 출력

SELECT EMPNO, ENAME, JOB FROM EMP WHERE DEPTNO = 10 OR DEPTNO = 30;

--11. 업무가 SALESMAN이거나 급여가 3000이상인 사람의 사번, 이름, 업무, 부서코드 출력

SELECT EMPNO, ENAME, JOB, SAL FROM EMP WHERE SAL > 3000 AND JOB = 'SALESMAN';

--12. 급여가 2500이상이고 업무가 MANAGER인 사람의 사번, 이름, 업무, 급여 출력

SELECT EMPNO, ENAME, JOB, SAL FROM EMP WHERE SAL > 2500 AND JOB = 'MANAGER';

13.“ename은 XXX 업무이고 연봉은 XX다” 스타일로 모두 출력(연봉은 SAL*12+COMM)


-- 8. SQL연산자(BETWEEN, IN, LIKE< IS NULL)

-- (1) BETWEEN A AND B : A부터 B까지(A, B포함. A<=B)
    -- ex1. SAL이 1500이상 3000이하
    SELECT * FROM EMP WHERE SAL >= 1500 AND SAL <= 3000;
    SELECT * FROM EMP WHERE SAL BETWEEN 1500 AND 3000;




