const students = [
  { id: 1, name: "Rahul", marks: 82 },
  { id: 2, name: "Priya", marks: 91 },
  { id: 3, name: "Anjali", marks: 68 },
  { id: 4, name: "Kiran", marks: 75 },
  { id: 5, name: "Ravi", marks: 55 }
];
//task 1
 console.log("task 1: display all students names using map()");

students.map(student => console.log(student.name));
//task 2
console.log("task 2 to find who score more than 75 using filter()");
const high= students.filter(student => student.marks > 75);
high.map(student => console.log(student.name));
//task 3
console.log("task 3 to find the average marks of all students using reduce()");
const totalMarks = students.reduce((sum, student) => {
  return sum + student.marks;
}, 0);
const average = totalMarks / students.length;
console.log("Average Marks", average);
//task 4
console.log("task 4 to find the student with id 4 using find()");
const student = students.find(student => student.id === 4);
console.log(student);
//task 5
console.log("task 5 to sort the students based on their marks in descending order using sort()");
const sortedStudents = students.sort((a, b) => {
  return b.marks - a.marks;
});
console.log(sortedStudents);